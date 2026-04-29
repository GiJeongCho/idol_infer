"""
RVC Voice Conversion - FastAPI Inference Server

Usage:
    cd /home/pps-nipa/NIQ/fish/side/idol/infer
    TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 python main.py
"""

import os
import sys

RVC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Retrieval-based-Voice-Conversion-WebUI")
RVC_ROOT = os.path.normpath(RVC_ROOT)

os.chdir(RVC_ROOT)
if RVC_ROOT not in sys.path:
    sys.path.insert(0, RVC_ROOT)

# -- PyTorch 2.6+ compatibility: patch torch.load before any model import --
import torch

_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": kwargs.get("weights_only", False)}
)

from dotenv import load_dotenv

load_dotenv(os.path.join(RVC_ROOT, ".env"))

import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from engine import RVCEngine
from trainer import RVCTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rvc-api")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine: Optional[RVCEngine] = None
trainer: Optional[RVCTrainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, trainer
    trainer = RVCTrainer(rvc_root=RVC_ROOT)
    logger.info("Trainer initialized")

    weight_dir = os.path.join(RVC_ROOT, os.getenv("weight_root", "assets/weights"))
    pth_files = [f for f in os.listdir(weight_dir) if f.endswith(".pth")]
    if pth_files:
        default_model = sorted(pth_files)[-1]
        logger.info("Loading RVC engine with model: %s", default_model)
        engine = RVCEngine(
            rvc_root=RVC_ROOT,
            model_name=default_model,
            device="cuda:0",
            is_half=False,
        )
        engine.load()
        logger.info("RVC engine ready")
    else:
        logger.info("No trained models found, skipping engine load (train first)")

    yield
    logger.info("Shutting down")
    if engine:
        engine.unload()


app = FastAPI(
    title="RVC Inference & Training API",
    version="3.0.0",
    lifespan=lifespan,
)


# -- utility endpoints --

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None and engine.ready}


@app.get("/models")
async def list_models():
    weight_dir = os.path.join(RVC_ROOT, os.getenv("weight_root", "assets/weights"))
    models = [f for f in os.listdir(weight_dir) if f.endswith(".pth")]
    return {"models": sorted(models), "current": engine.model_name if engine else None}


@app.post("/models/{model_name}")
async def switch_model(model_name: str):
    if engine is None:
        raise HTTPException(500, "Engine not initialized")
    try:
        engine.switch_model(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/outputs")
async def list_outputs():
    files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")],
        key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        reverse=True,
    )
    return {"output_dir": OUTPUT_DIR, "files": files}


@app.get("/outputs/{filename}")
async def download_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


# -- helper --

def _handle_model_switch(model: Optional[str]):
    global engine
    if model:
        if engine is None:
            logger.info("Lazy-loading engine with model: %s", model)
            engine = RVCEngine(
                rvc_root=RVC_ROOT,
                model_name=model,
                device="cuda:0",
                is_half=False,
            )
            engine.load()
        elif model != engine.model_name:
            try:
                engine.switch_model(model)
            except FileNotFoundError:
                raise HTTPException(404, f"Model not found: {model}")
            except Exception as e:
                raise HTTPException(400, f"Failed to switch model: {e}")


def _save_tmp(upload: UploadFile, content: bytes, label: str, run_id: str) -> str:
    suffix = os.path.splitext(upload.filename or "audio.wav")[1] or ".wav"
    path = os.path.join(tempfile.gettempdir(), f"rvc_{label}_{run_id}{suffix}")
    with open(path, "wb") as f:
        f.write(content)
    return path


# -- core API --

@app.post("/convert")
async def convert(
    vocal: UploadFile = File(..., description="vocals audio file"),
    bg: UploadFile = File(..., description="background music audio file"),
    model: Optional[str] = Form(None),
    f0_up_key: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.0),
    filter_radius: int = Form(3),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33),
    vocal_volume: float = Form(1.0),
    bg_volume: float = Form(1.0),
):
    """
    vocal(singing voice) + bg(background music) -> RVC converted vocal + bg mixed output

    - vocal: vocal/singing audio file
    - bg: background music/instrumental audio file
    - model: RVC model filename (e.g. Ralo_e200_s200.pth)
    - f0_up_key: pitch shift in semitones (-12 ~ +12)
    - f0_method: pitch extraction (rmvpe / pm / harvest / crepe)
    - index_rate: index mix ratio (0.0 ~ 1.0)
    - filter_radius: median filter radius for harvest
    - rms_mix_rate: RMS mix ratio (0.0 ~ 1.0)
    - protect: consonant protection (0.0 ~ 0.5)
    - vocal_volume: converted vocal volume multiplier
    - bg_volume: background music volume multiplier
    """
    _handle_model_switch(model)

    if engine is None or not engine.ready:
        raise HTTPException(503, "Model not loaded yet. Specify 'model' parameter.")

    run_id = uuid.uuid4().hex[:8]
    vocal_content = await vocal.read()
    bg_content = await bg.read()

    tmp_vocal = _save_tmp(vocal, vocal_content, "vocal", run_id)
    tmp_bg = _save_tmp(bg, bg_content, "bg", run_id)

    vocal_basename = os.path.splitext(vocal.filename or "vocal")[0]
    out_name = f"{vocal_basename}_{engine.model_name.replace('.pth','')}_{run_id}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        info = engine.convert_and_mix(
            vocal_path=tmp_vocal,
            bg_path=tmp_bg,
            output_path=out_path,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            vocal_volume=vocal_volume,
            bg_volume=bg_volume,
        )
        safe_info = info.replace("\n", " | ")
        return FileResponse(out_path, media_type="audio/wav", filename=out_name,
                            headers={"X-RVC-Info": safe_info})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Conversion error")
        raise HTTPException(500, str(e))
    finally:
        for p in (tmp_vocal, tmp_bg):
            if os.path.exists(p):
                os.remove(p)


# -- training API --

@app.post("/train")
async def start_training(
    exp_name: str = Form(..., description="Experiment name (folder under logs/)"),
    trainset_dir: str = Form(..., description="Absolute path to training audio folder"),
    sr: str = Form("48k", description="Target sample rate: 32k / 40k / 48k"),
    if_f0: bool = Form(True, description="Use F0 (pitch) guidance"),
    version: str = Form("v2", description="Model version: v1 / v2"),
    n_cpu: int = Form(4, description="CPU threads for preprocessing"),
    spk_id: int = Form(0, description="Speaker ID (0~4)"),
    f0_method: str = Form("rmvpe", description="F0 extraction: pm / harvest / dio / rmvpe / rmvpe_gpu"),
    gpus: str = Form("0", description="GPU IDs for training (dash-separated, e.g. 0-1)"),
    gpus_rmvpe: str = Form("0", description="GPU IDs for rmvpe_gpu (dash-separated)"),
    save_every_epoch: int = Form(10, description="Save checkpoint every N epochs (1~50)"),
    total_epoch: int = Form(100, description="Total training epochs (2~1000)"),
    batch_size: int = Form(8, description="Batch size per GPU (1~40)"),
    save_latest_only: bool = Form(False, description="Only keep latest checkpoint"),
    cache_gpu: bool = Form(False, description="Cache training set in GPU memory"),
    save_every_weights: bool = Form(True, description="Save small inference model at each save point"),
    pretrained_G: str = Form("", description="Pretrained Generator path (empty = auto)"),
    pretrained_D: str = Form("", description="Pretrained Discriminator path (empty = auto)"),
):
    """
    Start full RVC training pipeline (background job).

    Steps: preprocess -> extract_f0 -> extract_features -> train -> build_index

    Returns a job_id for status tracking via GET /train/{job_id}.
    """
    if trainer is None:
        raise HTTPException(500, "Trainer not initialized")

    job = trainer.start_train(
        exp_name=exp_name,
        trainset_dir=trainset_dir,
        sr=sr,
        if_f0=if_f0,
        version=version,
        n_cpu=n_cpu,
        spk_id=spk_id,
        f0_method=f0_method,
        gpus=gpus,
        gpus_rmvpe=gpus_rmvpe,
        save_every_epoch=save_every_epoch,
        total_epoch=total_epoch,
        batch_size=batch_size,
        save_latest_only=save_latest_only,
        cache_gpu=cache_gpu,
        save_every_weights=save_every_weights,
        pretrained_G=pretrained_G,
        pretrained_D=pretrained_D,
    )
    return {
        "message": "Training started",
        "job_id": job.job_id,
        "exp_name": exp_name,
        "status_url": f"/train/{job.job_id}",
    }


@app.get("/train/{job_id}")
async def get_train_status(job_id: str):
    """Get training job status and recent logs."""
    if trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    job = trainer.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job.to_dict()


@app.get("/train")
async def list_train_jobs():
    """List all training jobs."""
    if trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    jobs = [j.to_dict() for j in trainer.jobs.values()]
    jobs.sort(key=lambda j: j["elapsed_seconds"], reverse=True)
    return {"jobs": jobs}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=4998,
        log_level="info",
    )
