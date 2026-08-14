from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from qwen_asr import Qwen3ASRModel


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

MODEL_PATH = os.getenv("ASR_STREAM_MODEL", os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B"))
GPU_MEMORY_UTILIZATION = float(os.getenv("ASR_STREAM_GPU_MEMORY_UTILIZATION", "0.08"))
UNFIXED_CHUNK_NUM = int(os.getenv("ASR_STREAM_UNFIXED_CHUNK_NUM", "4"))
UNFIXED_TOKEN_NUM = int(os.getenv("ASR_STREAM_UNFIXED_TOKEN_NUM", "5"))
CHUNK_SIZE_SEC = float(os.getenv("ASR_STREAM_CHUNK_SIZE_SEC", "1.0"))
MAX_MODEL_LEN = int(os.getenv("ASR_STREAM_MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.getenv("ASR_STREAM_MAX_NUM_SEQS", "1"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("ASR_STREAM_MAX_NUM_BATCHED_TOKENS", "1024"))
TMP_DIR = Path(os.getenv("ASR_STREAM_TMP_DIR", "/tmp/meeting-ai-asr"))

app = FastAPI(title="Qwen ASR Low Latency Streaming", version="1.0.0")
asr: Qwen3ASRModel | None = None
asr_lock = threading.Lock()
sessions_lock = threading.Lock()
active_session_id: str | None = None


@dataclass
class Session:
    state: Any
    created_at: float
    last_seen: float


SESSIONS: dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60


def _gc_sessions() -> None:
    if asr is None:
        return
    now = time.time()
    with sessions_lock:
        dead = [sid for sid, item in SESSIONS.items() if now - item.last_seen > SESSION_TTL_SEC]
        for sid in dead:
            SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Session:
    _gc_sessions()
    with sessions_lock:
        session = SESSIONS.get(session_id)
        if not session:
            raise HTTPException(status_code=409, detail="session expired or replaced")
        if active_session_id and session_id != active_session_id:
            SESSIONS.pop(session_id, None)
            raise HTTPException(status_code=409, detail="session replaced")
        session.last_seen = time.time()
        return session


def _error_detail(exc: Exception, fallback: str) -> str:
    return str(exc).strip() or fallback


@app.on_event("startup")
def startup() -> None:
    global asr
    asr = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_new_tokens=32,
        enforce_eager=True,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    response_format: str = Form(default="json"),
    language: str | None = Form(default=None),
) -> dict[str, str]:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    target = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        forced_language = language.strip() if language else None
        with asr_lock:
            result = asr.transcribe(str(target), language=forced_language)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 文件转写失败，模型没有返回具体错误。")) from exc
    finally:
        target.unlink(missing_ok=True)

    return {
        "text": getattr(result, "text", "") or "",
        "language": getattr(result, "language", "") or "",
    }


@app.post("/api/start")
def start() -> dict[str, str]:
    global active_session_id
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    session_id = uuid.uuid4().hex
    try:
        state = asr.init_streaming_state(
            unfixed_chunk_num=UNFIXED_CHUNK_NUM,
            unfixed_token_num=UNFIXED_TOKEN_NUM,
            chunk_size_sec=CHUNK_SIZE_SEC,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 流式会话初始化失败。")) from exc
    now = time.time()
    with sessions_lock:
        SESSIONS.clear()
        SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
        active_session_id = session_id
    return {"session_id": session_id}


@app.post("/api/chunk")
async def chunk(request: Request, session_id: str = Query(...)) -> dict[str, str]:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    session = _get_session(session_id)
    if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
        raise HTTPException(status_code=400, detail="expect application/octet-stream")

    raw = await request.body()
    if len(raw) % 4 != 0:
        raise HTTPException(status_code=400, detail="float32 bytes length not multiple of 4")

    pcm = np.frombuffer(raw, dtype=np.float32).reshape(-1)
    try:
        with asr_lock:
            asr.streaming_transcribe(pcm, session.state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 流式识别失败。")) from exc
    return {
        "language": getattr(session.state, "language", "") or "",
        "text": getattr(session.state, "text", "") or "",
    }


@app.post("/api/finish")
def finish(session_id: str = Query(...)) -> dict[str, str]:
    global active_session_id
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    try:
        session = _get_session(session_id)
    except HTTPException as exc:
        if exc.status_code == 409:
            return {"language": "", "text": ""}
        raise
    try:
        with asr_lock:
            asr.finish_streaming_transcribe(session.state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 流式收尾失败。")) from exc
    output = {
        "language": getattr(session.state, "language", "") or "",
        "text": getattr(session.state, "text", "") or "",
    }
    with sessions_lock:
        SESSIONS.pop(session_id, None)
        if active_session_id == session_id:
            active_session_id = None
    return output
