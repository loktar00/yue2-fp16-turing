#!/usr/bin/env python3
"""YuE2 web UI + HTTP API (single file, FastAPI).

Start it with ./scripts/run.sh (or `uvicorn serve_yue2:app` from this folder).

* The port comes up immediately and nothing is loaded until the first request
  (or POST /load). The model is dropped again after YUE2_IDLE_UNLOAD_SECONDS of
  idleness, so the GPU is free when you are not generating.
* One job at a time; extra requests queue behind a lock.
* Every job writes FLAC (plus a PCM16 WAV the browser can play) to
  YUE2_OUT_DIR/<UTC timestamp>-<id>/.

API
  GET  /            web UI
  GET  /health      state, GPU, dtype, idle seconds
  POST /load        load the model now
  POST /unload      drop the model and free VRAM
  POST /v1/generate {"style","lyrics" | "instrumental": true,"seed","cot","id","cfg_scale"}
  GET  /v1/generate/{job}/audio?fmt=flac|wav
  GET  /v1/jobs     recent jobs

Environment (all optional)
  YUE2_DTYPE                 float16 (default) | bfloat16 | float32
  YUE2_OUT_DIR               ./outputs
  YUE2_GPU                   CUDA device index                               (0)
  YUE2_IDLE_UNLOAD_SECONDS   0 disables idle unload                          (600)
  YUE2_MEMORY_BUDGET_GIB     VRAM budget for the pipeline; default = card size
  YUE2_BACKEND               torch (CUDA graphs) | torch-eager               (torch)
  YUE2_OFFLOAD_AR            1 = park the AR model on CPU while decoding     (0)
  YUE2_VERIFY_HASHES         0 skips the sha256 check of the weights on load (1)
  YUE2_MODEL / YUE2_VAE      HF repo ids or local dirs  (m-a-p/YuE2-3B, m-a-p/YuE2-Vae)
  YUE2_HOST / YUE2_PORT      used by scripts/run.sh                          (127.0.0.1 / 5830)
"""
import gc
import json
import logging
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "YUE2_GPU" in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["YUE2_GPU"])

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("YUE2_OUT_DIR", HERE.parent / "outputs")).resolve()
MODEL_ID = os.environ.get("YUE2_MODEL", "m-a-p/YuE2-3B")
VAE_ID = os.environ.get("YUE2_VAE", "m-a-p/YuE2-Vae")
IDLE_UNLOAD_SECONDS = int(os.environ.get("YUE2_IDLE_UNLOAD_SECONDS", "600"))
BACKEND = os.environ.get("YUE2_BACKEND", "torch")
OFFLOAD_AR = os.environ.get("YUE2_OFFLOAD_AR", "0").lower() in ("1", "true", "yes")
VERIFY_HASHES = os.environ.get("YUE2_VERIFY_HASHES", "1").lower() not in ("0", "false", "no")
COT_CHOICES = ("full", "melody", "off")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("yue2")

STATE_LOCK = threading.RLock()
JOB_LOCK = threading.Lock()
_pipeline = None
_state = "unloaded"
_detail = "nothing loaded; the first request loads the model"
_last_activity = time.time()
_jobs_done = 0
_load_seconds = None


def _set(state, detail):
    global _state, _detail
    with STATE_LOCK:
        _state, _detail = state, detail


def _dtype_name():
    from yue2.dtype import model_dtype_name
    return model_dtype_name()


def _memory_budget_gib():
    env = os.environ.get("YUE2_MEMORY_BUDGET_GIB")
    if env:
        return float(env)
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return max(8.0, float(int(total)))


def _touch():
    global _last_activity
    _last_activity = time.time()


def ensure_loaded():
    """Build the pipeline if needed."""
    global _pipeline, _load_seconds
    with STATE_LOCK:
        if _pipeline is not None:
            return _pipeline
        t0 = time.time()
        _set("loading", "importing torch and yue2")
        try:
            import torch
            from yue2 import YuE2Pipeline
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available (is the NVIDIA driver installed and the CUDA build of torch in use?)")
            name = _dtype_name()
            log.info("loading on %s, dtype %s", torch.cuda.get_device_name(0), name)
            _set("loading", "resolving weights (the first run downloads ~7.3 GB)")
            pipe = YuE2Pipeline.from_pretrained(
                MODEL_ID, vae=VAE_ID, device="cuda", progress=False,
                verify_hashes=VERIFY_HASHES, memory_budget_gib=_memory_budget_gib(),
                backend=BACKEND, offload_ar=OFFLOAD_AR)
            _set("loading", "moving weights onto the GPU")
            pipe._load_model()
        except Exception as exc:
            log.error("load failed:\n%s", traceback.format_exc())
            _set("error", "%s: %s" % (type(exc).__name__, exc))
            raise
        _pipeline = pipe
        _load_seconds = round(time.time() - t0, 1)
        _touch()
        _set("ok", "loaded in %.1fs, dtype %s" % (_load_seconds, name))
        return pipe


def unload(reason="requested"):
    """Drop the pipeline and give the VRAM back (waits for a running job)."""
    global _pipeline
    with JOB_LOCK:
        with STATE_LOCK:
            pipe, _pipeline = _pipeline, None
            if pipe is None:
                return False
            try:
                pipe.close()
            except Exception as exc:
                log.warning("close() raised %s: %s", type(exc).__name__, exc)
            del pipe
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            _set("unloaded", "unloaded (%s)" % reason)
            log.info("unloaded (%s)", reason)
            return True


def _idle_watchdog():
    while True:
        time.sleep(15)
        if IDLE_UNLOAD_SECONDS <= 0 or _pipeline is None or JOB_LOCK.locked():
            continue
        if time.time() - _last_activity >= IDLE_UNLOAD_SECONDS:
            unload("idle %ds" % IDLE_UNLOAD_SECONDS)


def instrumental_lyrics(sections=4):
    """Any word in the lyrics field is sung, so an instrumental is bare tags only."""
    return "\n\n".join(["[Instrumental]"] * max(1, int(sections)))


def run_job(req):
    global _jobs_done
    song_id = req.get("id") or "song"
    if not SAFE_ID.fullmatch(song_id):
        raise ValueError("id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
    cot = req.get("cot", "full")
    if cot not in COT_CHOICES:
        raise ValueError("cot must be one of %s" % (COT_CHOICES,))
    if req.get("instrumental"):
        lyrics = instrumental_lyrics(req.get("sections", 4))
    else:
        lyrics = (req.get("lyrics") or "").strip()
    style = (req.get("style") or "").strip()
    if not style:
        raise ValueError("style is required")
    if not lyrics:
        raise ValueError("lyrics are required (or set instrumental=true)")

    kwargs = {"style": style, "lyrics": lyrics, "cot": cot, "seed": int(req.get("seed", 831001))}
    if req.get("cfg_scale") is not None:
        kwargs["cfg_scale"] = float(req["cfg_scale"])

    with JOB_LOCK:
        pipe = ensure_loaded()
        import soundfile as sf
        import torch
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_dir = OUT_DIR / ("%s-%s" % (stamp, song_id))
        job_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        _set("busy", "generating %s" % job_dir.name)
        t0 = time.time()
        try:
            song = pipe(**kwargs)
            saved = song.save_artifacts(str(job_dir))
            sf.write(str(job_dir / "audio.wav"), song.audio, song.sample_rate, subtype="PCM_16")
        finally:
            _set("ok", "idle, model resident")
            _touch()
        seconds = time.time() - t0
        duration = len(song.audio) / song.sample_rate
        result = {
            "id": song_id, "job": job_dir.name, "dir": str(job_dir),
            "audio_url": "/v1/generate/%s/audio" % job_dir.name,
            "flac_url": "/v1/generate/%s/audio?fmt=flac" % job_dir.name,
            "seed": kwargs["seed"], "dtype": _dtype_name(),
            "duration_s": round(duration, 1), "seconds": round(seconds, 1),
            "realtime_factor": round(seconds / duration, 3) if duration else None,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "truncated": song.truncated,
            "sample_rate": int(saved.get("sample_rate", song.sample_rate)),
        }
        (job_dir / "request.json").write_text(
            json.dumps(kwargs, indent=2, ensure_ascii=False), encoding="utf-8")
        del song
        torch.cuda.empty_cache()
        _jobs_done += 1
        log.info("job %s: %.1fs of audio in %.1fs, peak %.2f GiB",
                 job_dir.name, duration, seconds, result["peak_vram_gib"])
        return result


app = FastAPI(title="YuE2 (fp16, Turing)", version="1.0")


class GenerateRequest(BaseModel):
    style: str
    lyrics: str = ""
    instrumental: bool = False
    sections: int = 4
    id: str = "song"
    cot: str = "full"
    seed: int = 831001
    cfg_scale: float | None = None


def _gpu():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "-i", os.environ.get("YUE2_GPU", "0"),
             "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        name, total, used = [p.strip() for p in out.split(",")]
        return {"name": name, "total_mib": int(total), "used_mib": int(used)}
    except Exception:
        return None


@app.get("/health")
def health():
    with STATE_LOCK:
        return {"status": _state, "detail": _detail, "dtype": _dtype_name(), "model": MODEL_ID,
                "backend": BACKEND, "load_seconds": _load_seconds, "jobs_done": _jobs_done,
                "idle_seconds": round(time.time() - _last_activity, 1),
                "idle_unload_seconds": IDLE_UNLOAD_SECONDS, "gpu": _gpu(), "out_dir": str(OUT_DIR)}


def _safe(fn):
    try:
        fn()
    except Exception:
        pass


@app.post("/load")
def load():
    threading.Thread(target=_safe, args=(ensure_loaded,), daemon=True).start()
    return {"status": "loading"}


@app.post("/unload")
def unload_endpoint():
    return {"unloaded": unload("POST /unload")}


@app.post("/v1/generate")
def generate(req: GenerateRequest):
    try:
        return run_job(req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.error("job failed:\n%s", traceback.format_exc())
        raise HTTPException(500, "%s: %s" % (type(exc).__name__, exc))


def _job_dir(job):
    path = (OUT_DIR / job).resolve()
    if path.parent != OUT_DIR or not path.is_dir():
        raise HTTPException(404, "no such job")
    return path


@app.get("/v1/generate/{job}/audio")
def job_audio(job: str, fmt: str = "wav"):
    if fmt not in ("wav", "flac"):
        raise HTTPException(400, "fmt must be wav or flac")
    path = _job_dir(job) / ("audio." + fmt)
    if not path.is_file():
        raise HTTPException(404, "no audio")
    return FileResponse(path, media_type="audio/flac" if fmt == "flac" else "audio/wav",
                        filename="%s.%s" % (job, fmt))


@app.get("/v1/jobs")
def jobs(limit: int = 20):
    if not OUT_DIR.is_dir():
        return []
    dirs = sorted((d for d in OUT_DIR.iterdir() if (d / "audio.flac").is_file()), reverse=True)[:limit]
    return [{"job": d.name, "audio_url": "/v1/generate/%s/audio" % d.name,
             "flac_url": "/v1/generate/%s/audio?fmt=flac" % d.name} for d in dirs]


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


threading.Thread(target=_idle_watchdog, name="yue2-idle", daemon=True).start()
