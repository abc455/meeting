from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import MeetingCreate, TranscriptSegment


ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "db.json"
UPLOAD_DIR = STORAGE_DIR / "uploads"

_lock = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_db() -> dict[str, Any]:
    return {
        "meetings": [],
        "transcripts": {},
        "summaries": {},
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
    write_db(data)
    return meeting


def get_meeting(meeting_id: str) -> dict[str, Any] | None:
    data = read_db()
    return next((item for item in data["meetings"] if item["id"] == meeting_id), None)


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
