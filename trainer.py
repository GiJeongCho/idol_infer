"""
RVC Training Manager

Manages the full RVC training pipeline as background jobs:
  preprocess -> extract_f0 -> extract_features -> train -> build_index

All steps run as subprocesses (same as the WebUI) so GPU memory
is properly released between stages.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from random import shuffle
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger("rvc-trainer")

SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}
PYTHON = sys.executable


class TrainStatus(str, Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    EXTRACTING_F0 = "extracting_f0"
    EXTRACTING_FEATURES = "extracting_features"
    TRAINING = "training"
    BUILDING_INDEX = "building_index"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainJob:
    job_id: str
    exp_name: str
    status: TrainStatus = TrainStatus.PENDING
    progress: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    log_lines: list = field(default_factory=list)

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0
        end = self.finished_at or time.time()
        return end - self.started_at

    def add_log(self, line: str):
        self.log_lines.append(line)
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[-300:]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "exp_name": self.exp_name,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed(), 1),
            "log_tail": self.log_lines[-30:],
        }


class RVCTrainer:
    def __init__(self, rvc_root: str):
        self.rvc_root = rvc_root
        self.jobs: dict[str, TrainJob] = {}

    def start_train(
        self,
        exp_name: str,
        trainset_dir: str,
        sr: str = "48k",
        if_f0: bool = True,
        version: str = "v2",
        n_cpu: int = 4,
        spk_id: int = 0,
        f0_method: str = "rmvpe",
        gpus: str = "0",
        gpus_rmvpe: str = "0",
        save_every_epoch: int = 10,
        total_epoch: int = 100,
        batch_size: int = 8,
        save_latest_only: bool = False,
        cache_gpu: bool = False,
        save_every_weights: bool = True,
        pretrained_G: str = "",
        pretrained_D: str = "",
    ) -> TrainJob:
        job_id = uuid.uuid4().hex[:8]
        job = TrainJob(job_id=job_id, exp_name=exp_name)
        self.jobs[job_id] = job

        if not pretrained_G:
            pretrained_G = self._default_pretrained("G", sr, version, if_f0)
        if not pretrained_D:
            pretrained_D = self._default_pretrained("D", sr, version, if_f0)

        t = threading.Thread(
            target=self._run_pipeline,
            args=(job, trainset_dir, sr, if_f0, version, n_cpu, spk_id,
                  f0_method, gpus, gpus_rmvpe, save_every_epoch, total_epoch,
                  batch_size, save_latest_only, cache_gpu, save_every_weights,
                  pretrained_G, pretrained_D),
            daemon=True,
        )
        t.start()
        return job

    def get_job(self, job_id: str) -> Optional[TrainJob]:
        return self.jobs.get(job_id)

    # -- pipeline --

    def _run_pipeline(
        self, job, trainset_dir, sr, if_f0, version, n_cpu, spk_id,
        f0_method, gpus, gpus_rmvpe, save_every_epoch, total_epoch,
        batch_size, save_latest_only, cache_gpu, save_every_weights,
        pretrained_G, pretrained_D,
    ):
        job.started_at = time.time()
        exp_dir = os.path.join(self.rvc_root, "logs", job.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        sr_int = SR_MAP.get(sr, 48000)

        try:
            # Step 1: Preprocess
            job.status = TrainStatus.PREPROCESSING
            job.progress = "Step 1/5: Preprocessing audio"
            self._preprocess(job, trainset_dir, sr_int, n_cpu, exp_dir)

            # Step 2: Extract F0
            if if_f0:
                job.status = TrainStatus.EXTRACTING_F0
                job.progress = "Step 2/5: Extracting F0 (pitch)"
                self._extract_f0(job, exp_dir, f0_method, n_cpu, gpus_rmvpe)

            # Step 3: Extract features
            job.status = TrainStatus.EXTRACTING_FEATURES
            job.progress = "Step 3/5: Extracting HuBERT features"
            self._extract_features(job, exp_dir, gpus, version)

            # Step 4: Train
            job.status = TrainStatus.TRAINING
            job.progress = "Step 4/5: Training model"
            self._train(
                job, exp_dir, job.exp_name, sr, if_f0, spk_id, version,
                gpus, save_every_epoch, total_epoch, batch_size,
                save_latest_only, cache_gpu, save_every_weights,
                pretrained_G, pretrained_D,
            )

            # Step 5: Build index
            job.status = TrainStatus.BUILDING_INDEX
            job.progress = "Step 5/5: Building FAISS index"
            self._build_index(job, exp_dir, version)

            job.status = TrainStatus.COMPLETED
            job.progress = "Training completed"

        except Exception as e:
            job.status = TrainStatus.FAILED
            job.error = str(e)
            job.progress = f"Failed: {e}"
            logger.exception("Training failed for job %s", job.job_id)
        finally:
            job.finished_at = time.time()

    # -- individual steps --

    def _preprocess(self, job, trainset_dir, sr, n_cpu, exp_dir):
        cmd = (
            '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s" %s %.1f'
            % (PYTHON, trainset_dir, sr, n_cpu, exp_dir, "True", 3.7)
        )
        self._run_cmd_shell(job, cmd)

    def _extract_f0(self, job, exp_dir, f0_method, n_cpu, gpus_rmvpe):
        if f0_method == "rmvpe_gpu":
            gpu_list = gpus_rmvpe.split("-")
            for idx, gpu_id in enumerate(gpu_list):
                cmd = (
                    '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s" %s'
                    % (PYTHON, len(gpu_list), idx, gpu_id, exp_dir, "True")
                )
                self._run_cmd_shell(job, cmd)
        else:
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s" %s %s'
                % (PYTHON, exp_dir, n_cpu, f0_method)
            )
            self._run_cmd_shell(job, cmd)

    def _extract_features(self, job, exp_dir, gpus, version):
        gpu_list = gpus.split("-")
        n_parts = len(gpu_list)
        for idx, gpu_id in enumerate(gpu_list):
            cmd = (
                '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s" %s %s'
                % (PYTHON, f"cuda:{gpu_id}", n_parts, idx, gpu_id,
                   exp_dir, version, "False")
            )
            self._run_cmd_shell(job, cmd)

    def _train(
        self, job, exp_dir, exp_name, sr, if_f0, spk_id, version,
        gpus, save_every_epoch, total_epoch, batch_size,
        save_latest_only, cache_gpu, save_every_weights,
        pretrained_G, pretrained_D,
    ):
        self._write_filelist(exp_dir, sr, if_f0, spk_id, version)
        self._write_config_json(exp_dir, sr, version)

        f0_flag = 1 if if_f0 else 0
        latest_flag = 1 if save_latest_only else 0
        cache_flag = 1 if cache_gpu else 0
        weights_flag = 1 if save_every_weights else 0

        pg_part = '-pg "%s"' % pretrained_G if pretrained_G else ""
        pd_part = '-pd "%s"' % pretrained_D if pretrained_D else ""

        if gpus:
            cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
                % (PYTHON, exp_name, sr, f0_flag, batch_size, gpus,
                   total_epoch, save_every_epoch, pg_part, pd_part,
                   latest_flag, cache_flag, weights_flag, version)
            )
        else:
            cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
                % (PYTHON, exp_name, sr, f0_flag, batch_size,
                   total_epoch, save_every_epoch, pg_part, pd_part,
                   latest_flag, cache_flag, weights_flag, version)
            )

        self._run_cmd_shell(job, cmd)

        weight_dir = os.path.join(self.rvc_root, "assets", "weights")
        final_model = os.path.join(weight_dir, f"{exp_name}.pth")
        if not os.path.exists(final_model):
            any_model = any(
                f.startswith(exp_name) and f.endswith(".pth")
                for f in os.listdir(weight_dir)
            ) if os.path.isdir(weight_dir) else False
            if not any_model:
                raise RuntimeError(
                    "Training subprocess exited but no model was saved. "
                    "Check log for errors (e.g. NaN loss, OOM, missing files)."
                )

    def _build_index(self, job, exp_dir, version):
        feature_dir = os.path.join(
            exp_dir,
            "3_feature256" if version == "v1" else "3_feature768",
        )
        if not os.path.isdir(feature_dir):
            job.add_log("Feature directory not found, skipping index build")
            return

        npy_files = sorted(
            [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".npy")]
        )
        if not npy_files:
            job.add_log("No feature files found, skipping index build")
            return

        job.add_log(f"Loading {len(npy_files)} feature files")
        npys = [np.load(f) for f in npy_files]
        big_npy = np.concatenate(npys, axis=0)
        job.add_log(f"Total features shape: {big_npy.shape}")

        if big_npy.shape[0] > 200000:
            from sklearn.cluster import MiniBatchKMeans
            job.add_log("Large dataset, running k-means reduction to 10000 clusters")
            big_npy = MiniBatchKMeans(
                n_clusters=10000, verbose=True, batch_size=256 * os.cpu_count(),
                compute_labels=False, init="random",
            ).fit(big_npy).cluster_centers_

        np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

        dim = big_npy.shape[1]
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
        job.add_log(f"Training FAISS index (IVF{n_ivf},Flat, dim={dim})")
        index.train(big_npy)
        index.add(big_npy)

        index_path = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_1.index")
        faiss.write_index(index, index_path)
        job.add_log(f"Index saved: {index_path}")

    # -- helpers --

    def _write_config_json(self, exp_dir, sr, version):
        if version == "v1" or sr == "40k":
            config_file = f"v1/{sr}.json"
        else:
            config_file = f"v2/{sr}.json"

        src = os.path.join(self.rvc_root, "configs", config_file)
        inuse = os.path.join(self.rvc_root, "configs", "inuse", config_file)
        if not os.path.exists(inuse):
            os.makedirs(os.path.dirname(inuse), exist_ok=True)
            shutil.copy(src, inuse)

        config_save_path = os.path.join(exp_dir, "config.json")
        with open(inuse, "r") as f:
            cfg = json.load(f)
        cfg["train"]["fp16_run"] = False
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")
        logger.info("Config written (fp16 disabled): %s", config_save_path)

    def _write_filelist(self, exp_dir, sr, if_f0, spk_id, version):
        """Replicate infer-web.py click_train filelist generation exactly."""
        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        fea_dim = 256 if version == "v1" else 768
        feature_dir = os.path.join(exp_dir, f"3_feature{fea_dim}")

        if not os.path.isdir(gt_wavs_dir) or not os.path.isdir(feature_dir):
            raise FileNotFoundError("Preprocessed data not found. Run preprocess first.")

        if if_f0:
            f0_dir = os.path.join(exp_dir, "2a_f0")
            f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
            names = (
                set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
                & set(n.split(".")[0] for n in os.listdir(feature_dir))
                & set(n.split(".")[0] for n in os.listdir(f0_dir))
                & set(n.split(".")[0] for n in os.listdir(f0nsf_dir))
            )
        else:
            names = (
                set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
                & set(n.split(".")[0] for n in os.listdir(feature_dir))
            )

        now_dir = self.rvc_root
        opt = []
        for name in names:
            if if_f0:
                opt.append(
                    "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                    % (gt_wavs_dir, name, feature_dir, name,
                       os.path.join(exp_dir, "2a_f0"), name,
                       os.path.join(exp_dir, "2b-f0nsf"), name,
                       spk_id)
                )
            else:
                opt.append(
                    "%s/%s.wav|%s/%s.npy|%s"
                    % (gt_wavs_dir, name, feature_dir, name, spk_id)
                )

        if if_f0:
            for _ in range(2):
                opt.append(
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                    % (now_dir, sr, now_dir, fea_dim, now_dir, now_dir, spk_id)
                )
        else:
            for _ in range(2):
                opt.append(
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                    % (now_dir, sr, now_dir, fea_dim, spk_id)
                )

        shuffle(opt)
        with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
            f.write("\n".join(opt))

        logger.info("Filelist written: %d entries", len(opt))

    def _run_cmd_shell(self, job: TrainJob, cmd: str):
        """Run command via shell=True, same as WebUI's Popen(cmd, shell=True)."""
        job.add_log(f"$ {cmd}")
        logger.info("Running: %s", cmd)

        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.rvc_root,
        )
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                job.add_log(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed (exit {proc.returncode}): {cmd}")

    def _default_pretrained(self, g_or_d, sr, version, if_f0):
        prefix = "f0" if if_f0 else ""
        ver_dir = "pretrained_v2" if version == "v2" else "pretrained"
        filename = f"{prefix}{g_or_d}{sr}.pth"
        return f"assets/{ver_dir}/{filename}"
