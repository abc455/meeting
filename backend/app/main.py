from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .adapters import segments_from_text, stream_chunk, stream_finish, stream_start, summarize_meeting_cards, summarize_stage, summary_cards_to_text, transcribe_audio, write_minutes_document
from .schemas import AppSettings, AudioTranscriptionResult, Meeting, MeetingCreate, MeetingDetail, MinutesDocumentResult, StageSummary, StageSummaryResult, SummaryCards, SummaryResult, TranscriptSegment
from .store import (
    UPLOAD_DIR,
    create_meeting,
    delete_meeting,
    get_meeting,
    get_settings,
    read_db,
    replace_transcript,
    save_settings,
    save_minutes_document,
    save_summary,
    save_summary_cards,
    save_stage_summaries,
    update_meeting_status,
)


app = FastAPI(title="AI Meeting Transcription", version="1.0.0")
STREAM_STARTED_AT: dict[str, float] = {}
STREAM_CONTEXT: dict[str, dict] = {}

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




@app.get("/api/settings", response_model=AppSettings)
def get_app_settings() -> dict:
    return get_settings()


@app.put("/api/settings", response_model=AppSettings)
def put_app_settings(payload: AppSettings) -> dict:
    return save_settings(payload)


@app.get("/api/meetings", response_model=list[Meeting])
def list_meetings() -> list[dict]:
    return read_db()["meetings"]


@app.post("/api/meetings", response_model=Meeting)
def post_meeting(payload: MeetingCreate) -> dict:
    return create_meeting(payload)


@app.delete("/api/meetings/{meeting_id}")
def remove_meeting(meeting_id: str) -> dict[str, str]:
    if not delete_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"status": "deleted"}


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
        "summary_cards": data["summary_cards"].get(meeting_id, {}),
        "stage_summaries": data["stage_summaries"].get(meeting_id, []),
        "minutes_document": data["minutes_documents"].get(meeting_id, ""),
    }


def _save_upload(meeting_id: str, file: UploadFile, prefix: str) -> Path:
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    target = UPLOAD_DIR / f"{meeting_id}-{prefix}-{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return target


def _plain_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(item.text for item in segments if item.text).strip()


def _window_text(segments: list[TranscriptSegment], start: float, end: float) -> str:
    return "\n".join(
        item.text
        for item in segments
        if item.text and start <= ((item.start + item.end) / 2) < end
    ).strip()


def _stage_summary_reference(items: list[StageSummary]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"{item.start:.0f}-{item.end:.0f}秒：{item.title}")
        for summary in item.summary:
            lines.append(f"- {summary}")
    return "\n".join(lines).strip()


def _dirty_stage_summary(item: StageSummary) -> bool:
    text = "\n".join([item.title, *item.summary, *item.conclusions, *item.todos]).lower()
    return "thinking process" in text or "analyze user input" in text or "mental refinement" in text or "思考过程" in text


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
    save_summary(meeting_id, "")
    save_summary_cards(meeting_id, SummaryCards())
    save_stage_summaries(meeting_id, [])
    save_minutes_document(meeting_id, "")
    update_meeting_status(meeting_id, "转写完成")
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments)


@app.post("/api/meetings/{meeting_id}/summary", response_model=SummaryResult)
async def generate_summary(meeting_id: str) -> SummaryResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    data = read_db()
    transcript = data["transcripts"].get(meeting_id, [])
    text = _plain_text([TranscriptSegment(**item) for item in transcript])
    stage_summaries = [
        StageSummary(**item)
        for item in data["stage_summaries"].get(meeting_id, [])
        if item.get("summary") or item.get("conclusions") or item.get("todos")
    ]
    minutes_document = str(data["minutes_documents"].get(meeting_id, "") or "").strip()
    try:
        if not minutes_document:
            minutes_document = await write_minutes_document(meeting["title"], text, _stage_summary_reference(stage_summaries))
            save_minutes_document(meeting_id, minutes_document)
        summary_cards = await summarize_meeting_cards(meeting["title"], text, minutes_document)
        summary = summary_cards_to_text(summary_cards)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "智能纪要生成失败，后端没有返回具体错误。")) from exc
    save_summary(meeting_id, summary)
    save_summary_cards(meeting_id, summary_cards)
    update_meeting_status(meeting_id, "纪要已生成")
    return SummaryResult(summary=summary, summary_cards=summary_cards, minutes_document=minutes_document)


@app.post("/api/meetings/{meeting_id}/stage-summaries", response_model=StageSummaryResult)
async def generate_stage_summaries(
    meeting_id: str,
    window_seconds: int = Query(120, ge=30, le=600),
    refresh: bool = Query(False),
) -> StageSummaryResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    data = read_db()
    segments = [
        TranscriptSegment(**item)
        for item in data["transcripts"].get(meeting_id, [])
        if item.get("text")
    ]
    if not segments:
        raise HTTPException(status_code=400, detail="当前会议还没有转写文字，无法生成阶段摘要。")

    existing = [
        StageSummary(**item)
        for item in data["stage_summaries"].get(meeting_id, [])
        if item.get("summary") or item.get("conclusions") or item.get("todos")
    ]
    existing = [item for item in existing if not _dirty_stage_summary(item)]
    if refresh:
        existing = []
    existing_windows = {(round(item.start, 1), round(item.end, 1)) for item in existing}
    max_end = max(item.end for item in segments)
    cursor = 0.0

    try:
        while cursor < max_end:
            end = min(cursor + float(window_seconds), max_end)
            text = _window_text(segments, cursor, end)
            window_key = (round(cursor, 1), round(end, 1))
            if text and window_key not in existing_windows:
                result = await summarize_stage(meeting["title"], cursor, end, text)
                existing.append(
                    StageSummary(
                        id=uuid.uuid4().hex,
                        start=round(cursor, 2),
                        end=round(end, 2),
                        title=result["title"],
                        summary=result["summary"],
                        conclusions=result["conclusions"],
                        todos=result["todos"],
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                )
                existing_windows.add(window_key)
            cursor += float(window_seconds)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "阶段摘要生成失败，后端没有返回具体错误。")) from exc

    existing.sort(key=lambda item: item.start)
    save_stage_summaries(meeting_id, existing)
    return StageSummaryResult(stage_summaries=existing)


@app.post("/api/meetings/{meeting_id}/minutes-document", response_model=MinutesDocumentResult)
async def generate_minutes_document(meeting_id: str) -> MinutesDocumentResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    data = read_db()
    segments = [
        TranscriptSegment(**item)
        for item in data["transcripts"].get(meeting_id, [])
        if item.get("text")
    ]
    text = _plain_text(segments)
    stage_summaries = [
        StageSummary(**item)
        for item in data["stage_summaries"].get(meeting_id, [])
        if item.get("summary") or item.get("conclusions") or item.get("todos")
    ]
    try:
        document = await write_minutes_document(meeting["title"], text, _stage_summary_reference(stage_summaries))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "正式纪要生成失败，后端没有返回具体错误。")) from exc

    save_minutes_document(meeting_id, document)
    update_meeting_status(meeting_id, "正式纪要已生成")
    return MinutesDocumentResult(document=document)


@app.post("/api/meetings/{meeting_id}/stream/start")
async def start_low_latency_stream(meeting_id: str) -> dict[str, str]:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        session_id = await stream_start()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 启动失败，后端没有返回具体错误。")) from exc
    existing = [
        TranscriptSegment(**item)
        for item in read_db()["transcripts"].get(meeting_id, [])
        if item.get("text")
    ]
    offset = max((item.end for item in existing), default=0)
    STREAM_STARTED_AT[session_id] = time.monotonic()
    STREAM_CONTEXT[session_id] = {
        "meeting_id": meeting_id,
        "offset": offset,
        "prefix": existing,
    }
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
    context = STREAM_CONTEXT.get(session_id, {})
    prefix = context.get("prefix") or []
    offset = float(context.get("offset") or 0)
    duration = time.monotonic() - STREAM_STARTED_AT.get(session_id, time.monotonic())
    current_segments = segments_from_text(text, language=payload.get("language") or "zh", offset=offset, duration=duration)
    segments = [*prefix, *current_segments]
    replace_transcript(meeting_id, segments)
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments)


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
    context = STREAM_CONTEXT.pop(session_id, {})
    prefix = context.get("prefix") or []
    offset = float(context.get("offset") or 0)
    duration = time.monotonic() - STREAM_STARTED_AT.pop(session_id, time.monotonic())
    current_segments = segments_from_text(text, language=payload.get("language") or "zh", offset=offset, duration=duration)
    segments = [*prefix, *current_segments]
    replace_transcript(meeting_id, segments)
    update_meeting_status(meeting_id, "转写完成")
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments)
