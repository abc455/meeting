from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import AppSettings, MeetingCreate, StageSummary, SummaryCards, TranscriptSegment


ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "db.json"
UPLOAD_DIR = STORAGE_DIR / "uploads"

_lock = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_settings() -> dict[str, Any]:
    return AppSettings(
        asr_base_url=os.getenv("ASR_BASE_URL", ""),
        asr_model=os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B"),
        asr_api_key=os.getenv("ASR_API_KEY", "EMPTY"),
        asr_transcribe_path=os.getenv("ASR_TRANSCRIBE_PATH", "/v1/audio/transcriptions"),
        asr_max_retries=int(os.getenv("ASR_MAX_RETRIES", "2")),
        asr_stream_base_url=os.getenv("ASR_STREAM_BASE_URL", "http://127.0.0.1:8006"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-NVFP4"),
        llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
    ).model_dump()


def empty_db() -> dict[str, Any]:
    return {
        "meetings": [],
        "transcripts": {},
        "summaries": {},
        "summary_cards": {},
        "stage_summaries": {},
        "minutes_documents": {},
        "settings": default_settings(),
    }


def normalize_db(data: dict[str, Any]) -> dict[str, Any]:
    clean = empty_db()
    clean["meetings"] = [
        {
            "id": str(item["id"]),
            "title": str(item.get("title") or "未命名会议"),
            "status": str(item.get("status") or "已创建"),
            "created_at": str(item.get("created_at") or now_iso()),
        }
        for item in data.get("meetings", [])
        if item.get("id")
    ]
    for meeting in clean["meetings"]:
        meeting_id = meeting["id"]
        clean["transcripts"][meeting_id] = [
            TranscriptSegment(**segment).model_dump()
            for segment in data.get("transcripts", {}).get(meeting_id, [])
            if segment.get("text")
        ]
        clean["summaries"][meeting_id] = str(data.get("summaries", {}).get(meeting_id, ""))
        clean["summary_cards"][meeting_id] = SummaryCards(**(data.get("summary_cards", {}).get(meeting_id, {}) or {})).model_dump()
        clean["stage_summaries"][meeting_id] = [
            StageSummary(**item).model_dump()
            for item in data.get("stage_summaries", {}).get(meeting_id, [])
            if item.get("summary") or item.get("conclusions") or item.get("todos")
        ]
        clean["minutes_documents"][meeting_id] = str(data.get("minutes_documents", {}).get(meeting_id, ""))

    settings = default_settings()
    settings.update(data.get("settings", {}) or {})
    clean["settings"] = AppSettings(**settings).model_dump()
    return clean


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        write_db(empty_db())


def read_db() -> dict[str, Any]:
    ensure_storage()
    with _lock:
        return normalize_db(json.loads(DB_PATH.read_text(encoding="utf-8")))


def write_db(data: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        DB_PATH.write_text(json.dumps(normalize_db(data), ensure_ascii=False, indent=2), encoding="utf-8")


def create_meeting(payload: MeetingCreate) -> dict[str, Any]:
    data = read_db()
    meeting = {
        "id": uuid.uuid4().hex,
        "title": payload.title.strip(),
        "status": "已创建",
        "created_at": now_iso(),
    }
    data["meetings"].insert(0, meeting)
    data["transcripts"][meeting["id"]] = []
    data["summaries"][meeting["id"]] = ""
    data["summary_cards"][meeting["id"]] = SummaryCards().model_dump()
    data["stage_summaries"][meeting["id"]] = []
    data["minutes_documents"][meeting["id"]] = ""
    write_db(data)
    return meeting


def get_meeting(meeting_id: str) -> dict[str, Any] | None:
    data = read_db()
    return next((item for item in data["meetings"] if item["id"] == meeting_id), None)


def delete_meeting(meeting_id: str) -> bool:
    data = read_db()
    before_count = len(data["meetings"])
    data["meetings"] = [item for item in data["meetings"] if item["id"] != meeting_id]
    if len(data["meetings"]) == before_count:
        return False

    data["transcripts"].pop(meeting_id, None)
    data["summaries"].pop(meeting_id, None)
    data["summary_cards"].pop(meeting_id, None)
    data["stage_summaries"].pop(meeting_id, None)
    data["minutes_documents"].pop(meeting_id, None)
    write_db(data)

    for file_path in UPLOAD_DIR.glob(f"{meeting_id}-*"):
        if file_path.is_file():
            file_path.unlink(missing_ok=True)
    return True


def update_meeting_status(meeting_id: str, status: str) -> None:
    data = read_db()
    for meeting in data["meetings"]:
        if meeting["id"] == meeting_id:
            meeting["status"] = status
            break
    write_db(data)


def replace_transcript(meeting_id: str, segments: list[TranscriptSegment]) -> None:
    data = read_db()
    data["transcripts"][meeting_id] = [item.model_dump() for item in segments]
    write_db(data)


def save_summary(meeting_id: str, summary: str) -> None:
    data = read_db()
    data["summaries"][meeting_id] = summary
    write_db(data)


def save_summary_cards(meeting_id: str, summary_cards: SummaryCards) -> None:
    data = read_db()
    data["summary_cards"][meeting_id] = summary_cards.model_dump()
    write_db(data)


def save_stage_summaries(meeting_id: str, stage_summaries: list[StageSummary]) -> None:
    data = read_db()
    data["stage_summaries"][meeting_id] = [item.model_dump() for item in stage_summaries]
    write_db(data)


def save_minutes_document(meeting_id: str, document: str) -> None:
    data = read_db()
    data["minutes_documents"][meeting_id] = document
    write_db(data)


def get_settings() -> dict[str, Any]:
    return read_db()["settings"]


def save_settings(settings: AppSettings) -> dict[str, Any]:
    data = read_db()
    data["settings"] = settings.model_dump()
    write_db(data)
    return data["settings"]
