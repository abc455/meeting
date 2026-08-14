from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .adapters import stream_chunk, stream_finish, stream_start, summarize_meeting, transcribe_audio
from .schemas import AudioTranscriptionResult, Meeting, MeetingCreate, MeetingDetail, SummaryResult, TranscriptSegment
from .store import (
    UPLOAD_DIR,
    create_meeting,
    get_meeting,
    read_db,
    replace_transcript,
    save_summary,
    update_meeting_status,
)


app = FastAPI(title="AI Meeting Transcription", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meetings", response_model=list[Meeting])
def list_meetings() -> list[dict]:
    return read_db()["meetings"]


@app.post("/api/meetings", response_model=Meeting)
def post_meeting(payload: MeetingCreate) -> dict:
    return create_meeting(payload)


@app.get("/api/meetings/{meeting_id}", response_model=MeetingDetail)
def get_meeting_detail(meeting_id: str) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    data = read_db()
    return {
        "meeting": meeting,
        "transcript": data["transcripts"].get(meeting_id, []),
        "summary": data["summaries"].get(meeting_id, ""),
    }


def _save_upload(meeting_id: str, file: UploadFile, prefix: str) -> Path:
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    target = UPLOAD_DIR / f"{meeting_id}-{prefix}-{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return target


def _plain_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(item.text for item in segments if item.text).strip()


def _error_detail(exc: Exception, fallback: str) -> str:
    return str(exc).strip() or fallback


@app.post("/api/meetings/{meeting_id}/audio", response_model=AudioTranscriptionResult)
async def upload_audio(meeting_id: str, file: UploadFile = File(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")

    target = _save_upload(meeting_id, file, "upload")
    update_meeting_status(meeting_id, "转写中")
    try:
        segments = await transcribe_audio(target)
    except Exception as exc:
        update_meeting_status(meeting_id, "转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "音频转写失败，后端没有返回具体错误。")) from exc

    replace_transcript(meeting_id, segments)
    update_meeting_status(meeting_id, "转写完成")
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments)


@app.post("/api/meetings/{meeting_id}/summary", response_model=SummaryResult)
async def generate_summary(meeting_id: str) -> SummaryResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    transcript = read_db()["transcripts"].get(meeting_id, [])
    text = _plain_text([TranscriptSegment(**item) for item in transcript])
    try:
        summary = await summarize_meeting(meeting["title"], text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "AI 纪要生成失败，后端没有返回具体错误。")) from exc
    save_summary(meeting_id, summary)
    update_meeting_status(meeting_id, "纪要已生成")
    return SummaryResult(summary=summary)


@app.post("/api/meetings/{meeting_id}/stream/start")
async def start_low_latency_stream(meeting_id: str) -> dict[str, str]:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        session_id = await stream_start()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 启动失败，后端没有返回具体错误。")) from exc
    replace_transcript(meeting_id, [])
    update_meeting_status(meeting_id, "实时转写中")
    return {"session_id": session_id}


@app.post("/api/meetings/{meeting_id}/stream/chunk", response_model=AudioTranscriptionResult)
async def push_low_latency_stream_chunk(
    meeting_id: str,
    request: Request,
    session_id: str = Query(...),
) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        payload = await stream_chunk(session_id, await request.body())
    except Exception as exc:
        update_meeting_status(meeting_id, "实时转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 识别失败，后端没有返回具体错误。")) from exc

    text = payload["text"].strip()
    segments = [
        TranscriptSegment(id=session_id, start=0, end=0, text=text, language=payload.get("language") or "zh")
    ] if text else []
    replace_transcript(meeting_id, segments)
    return AudioTranscriptionResult(text=text, segments=segments)


@app.post("/api/meetings/{meeting_id}/stream/finish", response_model=AudioTranscriptionResult)
async def finish_low_latency_stream(meeting_id: str, session_id: str = Query(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        payload = await stream_finish(session_id)
    except Exception as exc:
        update_meeting_status(meeting_id, "实时转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 收尾失败，后端没有返回具体错误。")) from exc

    text = payload["text"].strip()
    segments = [
        TranscriptSegment(id=session_id, start=0, end=0, text=text, language=payload.get("language") or "zh")
    ] if text else []
    replace_transcript(meeting_id, segments)
    update_meeting_status(meeting_id, "转写完成")
    return AudioTranscriptionResult(text=text, segments=segments)
