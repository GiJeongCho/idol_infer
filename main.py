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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rvc-api")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine: Optional[RVCEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Loading RVC engine")
    engine = RVCEngine(
        rvc_root=RVC_ROOT,
        model_name="Ralo_e200_s200.pth",
        device="cuda:0",
        is_half=False,
    )
    engine.load()
    logger.info("RVC engine ready")
    yield
    logger.info("Shutting down RVC engine")
    engine.unload()


app = FastAPI(
    title="RVC Inference API",
    version="2.0.0",
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
    if model and model != engine.model_name:
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
    if engine is None or not engine.ready:
        raise HTTPException(503, "Model not loaded yet")

    _handle_model_switch(model)

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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=4998,
        log_level="info",
    )
