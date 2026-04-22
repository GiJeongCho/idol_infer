"""
RVC Inference Engine — standalone wrapper around the RVC pipeline.

Handles model loading, compatibility patches, and voice conversion
without depending on the Gradio WebUI.

Supports automatic vocal separation (demucs) so that background music
is preserved and only the vocal track gets converted.
"""

import logging
import os
import sys
import traceback

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("rvc-engine")

from infer.lib.audio import load_audio
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from fairseq import checkpoint_utils


class _Config:
    """Minimal config matching what Pipeline/VC expects, without argparse."""

    def __init__(self, device: str = "cuda:0", is_half: bool = False):
        self.device = device
        self.is_half = is_half
        self.use_jit = False

        if is_half:
            self.x_pad = 3
            self.x_query = 10
            self.x_center = 60
            self.x_max = 65
        else:
            self.x_pad = 1
            self.x_query = 6
            self.x_center = 38
            self.x_max = 41


class RVCEngine:
    def __init__(
        self,
        rvc_root: str,
        model_name: str = "Ralo_e200_s200.pth",
        device: str = "cuda:0",
        is_half: bool = False,
    ):
        self.rvc_root = rvc_root
        self.model_name = model_name
        self.device = device
        self.is_half = is_half

        self.config = _Config(device=device, is_half=is_half)
        self.hubert_model = None
        self.net_g = None
        self.pipeline = None
        self.tgt_sr = None
        self.if_f0 = 1
        self.version = "v2"
        self.cpt = None
        self.ready = False

    # ── public API ────────────────────────────────────────────────────────

    def load(self):
        """Load Hubert + RVC model, build pipeline."""
        self._load_hubert()
        self._load_rvc_model(self.model_name)
        self.ready = True
        logger.info(
            "Engine ready — model=%s  tgt_sr=%d  version=%s  if_f0=%d  device=%s  half=%s",
            self.model_name,
            self.tgt_sr,
            self.version,
            self.if_f0,
            self.device,
            self.is_half,
        )

    def unload(self):
        for attr in ("net_g", "hubert_model", "pipeline"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.ready = False

    def switch_model(self, model_name: str):
        weight_root = os.getenv("weight_root", "assets/weights")
        path = os.path.join(weight_root, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found: {path}")

        if self.net_g is not None:
            del self.net_g
            self.net_g = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._load_rvc_model(model_name)
        logger.info("Switched to model: %s", model_name)

    def convert(
        self,
        input_audio_path: str,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        file_index: str = "",
        index_rate: float = 0.0,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        f0_file=None,
    ) -> tuple:
        """
        Run voice conversion.

        Returns (info_str, (sample_rate, audio_int16_ndarray)).
        On failure returns (error_str, (None, None)).
        """
        if not self.ready:
            return "Engine not loaded", (None, None)

        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            times = [0, 0, 0]

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                0,  # sid (speaker id)
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )

            tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr

            index_info = (
                f"Index: {file_index}"
                if file_index and os.path.exists(file_index)
                else "Index not used."
            )
            info = (
                f"Success.\n{index_info}\n"
                f"Time: npy={times[0]:.2f}s, f0={times[1]:.2f}s, infer={times[2]:.2f}s"
            )
            return info, (tgt_sr, audio_opt)
        except Exception:
            err = traceback.format_exc()
            logger.error("Conversion failed:\n%s", err)
            return err, (None, None)

    def convert_and_mix(
        self,
        vocal_path: str,
        bg_path: str,
        output_path: str,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        file_index: str = "",
        index_rate: float = 0.0,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        vocal_volume: float = 1.0,
        bg_volume: float = 1.0,
    ) -> str:
        """
        vocal + bg -> RVC(vocal) + bg mixed output.

        1. RVC converts the vocal track
        2. Mix converted vocal with background music

        Returns info string on success, raises on failure.
        """
        if not self.ready:
            raise RuntimeError("Engine not loaded")

        logger.info("Step 1/2: Converting vocals with RVC")
        info, (sr, converted_vocal) = self.convert(
            input_audio_path=vocal_path,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            file_index=file_index,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=0,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        if sr is None or converted_vocal is None:
            raise RuntimeError(f"RVC conversion failed: {info}")

        logger.info("Step 2/2: Mixing converted vocals with background")
        self._remix(converted_vocal, sr, bg_path, output_path, vocal_volume, bg_volume)

        logger.info("Done -> %s", output_path)
        return info

    # -- internals --

    def _remix(
        self,
        converted_vocal: np.ndarray,
        vocal_sr: int,
        bg_path: str,
        output_path: str,
        vocal_volume: float = 1.0,
        bg_volume: float = 1.0,
    ):
        """Mix converted vocals (int16) with background audio and save."""
        import librosa

        bg_audio, bg_sr = sf.read(bg_path, dtype="float32")
        if bg_audio.ndim == 2:
            bg_audio = bg_audio.mean(axis=1)

        if bg_sr != vocal_sr:
            bg_audio = librosa.resample(bg_audio, orig_sr=bg_sr, target_sr=vocal_sr)

        vocal_float = converted_vocal.astype(np.float32) / 32768.0

        min_len = min(len(vocal_float), len(bg_audio))
        vocal_float = vocal_float[:min_len]
        bg_audio = bg_audio[:min_len]

        mixed = vocal_float * vocal_volume + bg_audio * bg_volume

        peak = np.abs(mixed).max()
        if peak > 0.99:
            mixed *= 0.99 / peak

        sf.write(output_path, mixed, vocal_sr)

    def _load_hubert(self):
        hubert_path = os.path.join(self.rvc_root, "assets", "hubert", "hubert_base.pt")
        logger.info("Loading Hubert from %s", hubert_path)

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path], suffix=""
        )
        hubert = models[0].to(self.device)
        hubert = hubert.half() if self.is_half else hubert.float()
        self.hubert_model = hubert.eval()
        logger.info("Hubert loaded")

    def _load_rvc_model(self, model_name: str):
        weight_root = os.getenv("weight_root", "assets/weights")
        model_path = os.path.join(weight_root, model_name)
        logger.info("Loading RVC model from %s", model_path)

        cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v1")
        self.cpt = cpt

        cls_map = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }
        cls = cls_map.get((self.version, self.if_f0), SynthesizerTrnMs256NSFsid)
        self.net_g = cls(*cpt["config"], is_half=self.is_half)
        del self.net_g.enc_q

        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        self.net_g = self.net_g.half() if self.is_half else self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        self.model_name = model_name
        logger.info("RVC model loaded: %s (sr=%d, version=%s, f0=%d)", model_name, self.tgt_sr, self.version, self.if_f0)
