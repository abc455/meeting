from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .schemas import SummaryCards, SummaryConclusion, SummaryDetailGroup, SummaryRisk, SummaryTodo, TranscriptSegment


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ASR_BASE_URL = os.getenv("ASR_BASE_URL", "").rstrip("/")
ASR_API_KEY = os.getenv("ASR_API_KEY", "EMPTY")
ASR_MODEL = os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B")
ASR_TRANSCRIBE_PATH = os.getenv("ASR_TRANSCRIBE_PATH", "/v1/audio/transcriptions")
ASR_MAX_RETRIES = int(os.getenv("ASR_MAX_RETRIES", "2"))
ASR_STREAM_BASE_URL = os.getenv("ASR_STREAM_BASE_URL", "http://127.0.0.1:8006").rstrip("/")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-NVFP4")

SENTENCE_END_RE = re.compile(r"[。！？!?]$")
SOFT_BREAK_RE = re.compile(r"[，,、；;：:]$")
TEXT_UNIT_RE = re.compile(r"[^。！？!?；;，,、：:\n]+[。！？!?；;，,、：:]?")
MIN_SEGMENT_CHARS = 18
TARGET_SEGMENT_CHARS = 42
MAX_SEGMENT_CHARS = 86
TIMESTAMP_PAUSE_BREAK_SEC = 1.2
SEGMENTER_ENGINE = os.getenv("ASR_SEGMENTER", "heuristic").strip().lower()


def _settings() -> dict:
    from .store import get_settings

    return get_settings()


def _require_asr(settings: dict) -> str:
    base_url = str(settings.get("asr_base_url") or ASR_BASE_URL).rstrip("/")
    path = str(settings.get("asr_transcribe_path") or ASR_TRANSCRIBE_PATH)
    if not base_url:
        raise RuntimeError("未配置 ASR_BASE_URL，无法进行真实转写。")
    return f"{base_url}{path}"


def _segment_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _normalize_asr_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def _split_long_unit(text: str, max_len: int = MAX_SEGMENT_CHARS) -> list[str]:
    if _segment_len(text) <= max_len:
        return [text]

    pieces: list[str] = []
    current = ""
    for unit in TEXT_UNIT_RE.findall(text) or [text]:
        unit = unit.strip()
        if not unit:
            continue
        if current and _segment_len(current + unit) > max_len:
            pieces.append(current.strip())
            current = unit
        else:
            current += unit
    if current.strip():
        pieces.append(current.strip())

    chunks: list[str] = []
    for piece in pieces:
        while _segment_len(piece) > max_len:
            chunks.append(piece[:max_len].strip())
            piece = piece[max_len:].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _split_with_funasr_placeholder(text: str) -> list[str] | None:
    # Reserved for FunASR VAD + punctuation based segmentation.
    # Future hook: return sentence/paragraph chunks from FunASR instead of the heuristic splitter.
    return None


def _split_transcript_text(text: str) -> list[str]:
    text = _clean_asr_text(text)
    if not text:
        return []

    text = _normalize_asr_spacing(text)
    if SEGMENTER_ENGINE == "funasr":
        funasr_segments = _split_with_funasr_placeholder(text)
        if funasr_segments:
            return funasr_segments

    raw_units = [item.strip() for item in TEXT_UNIT_RE.findall(text) if item.strip()] or [text]
    units: list[str] = []
    for raw_unit in raw_units:
        units.extend(_split_long_unit(raw_unit))

    segments: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue

        current_len = _segment_len(current)
        next_len = _segment_len(unit)
        combined_len = _segment_len(current + unit)
        current_has_sentence_end = bool(SENTENCE_END_RE.search(current))
        current_has_soft_break = bool(SOFT_BREAK_RE.search(current))

        should_flush = (
            combined_len > MAX_SEGMENT_CHARS
            or (current_has_sentence_end and current_len >= MIN_SEGMENT_CHARS)
            or (current_has_soft_break and current_len >= TARGET_SEGMENT_CHARS and next_len >= 6)
        )
        if should_flush:
            segments.append(current.strip())
            current = unit
        else:
            current += unit

    if current.strip():
        segments.append(current.strip())

    if len(segments) >= 2 and _segment_len(segments[-1]) < MIN_SEGMENT_CHARS:
        tail = segments.pop()
        segments[-1] = f"{segments[-1]}{tail}"

    return segments


def _duration_seconds(file_path: Path) -> float | None:
    try:
        with wave.open(str(file_path), "rb") as audio:
            rate = audio.getframerate()
            if rate <= 0:
                return None
            return audio.getnframes() / float(rate)
    except Exception:
        return None


def segments_from_text(
    text: str,
    language: str = "zh",
    offset: float = 0,
    duration: float | None = None,
) -> list[TranscriptSegment]:
    units = _split_transcript_text(text)
    if not units:
        return []

    if duration and duration > 0:
        total_chars = sum(max(1, len(item)) for item in units)
        cursor = float(offset)
        segments: list[TranscriptSegment] = []
        for item in units:
            span = max(0.75, duration * max(1, len(item)) / max(1, total_chars))
            start = cursor
            end = min(offset + duration, cursor + span)
            segments.append(
                TranscriptSegment(
                    id=uuid.uuid4().hex,
                    start=round(start, 2),
                    end=round(max(end, start + 0.5), 2),
                    text=item,
                    language=language,
                )
            )
            cursor = end
        return segments

    return [
        TranscriptSegment(
            id=uuid.uuid4().hex,
            start=round(offset + index * 3.0, 2),
            end=round(offset + index * 3.0 + 2.5, 2),
            text=item,
            language=language,
        )
        for index, item in enumerate(units)
    ]


def _merge_transcript_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    current: TranscriptSegment | None = None

    for segment in segments:
        text = _normalize_asr_spacing(segment.text)
        if not text:
            continue
        item = segment.model_copy(update={"text": text})
        if current is None:
            current = item
            continue

        gap = max(0.0, item.start - current.end)
        current_len = _segment_len(current.text)
        combined_text = f"{current.text}{item.text}"
        combined_len = _segment_len(combined_text)
        should_flush = (
            gap >= TIMESTAMP_PAUSE_BREAK_SEC and current_len >= MIN_SEGMENT_CHARS
        ) or (
            SENTENCE_END_RE.search(current.text) and current_len >= MIN_SEGMENT_CHARS
        ) or (
            combined_len > MAX_SEGMENT_CHARS
        )

        if should_flush:
            merged.append(current)
            current = item
        else:
            current = current.model_copy(
                update={
                    "end": max(current.end, item.end),
                    "text": combined_text,
                    "language": current.language or item.language,
                }
            )

    if current is not None:
        merged.append(current)
    if len(merged) >= 2 and _segment_len(merged[-1].text) < MIN_SEGMENT_CHARS:
        tail = merged.pop()
        previous = merged[-1]
        merged[-1] = previous.model_copy(
            update={
                "end": max(previous.end, tail.end),
                "text": f"{previous.text}{tail.text}",
            }
        )
    return merged


def _timestamp_segments_from_payload(payload: dict, language: str, offset: float = 0) -> list[TranscriptSegment]:
    raw_segments = (
        payload.get("segments")
        or payload.get("time_stamps")
        or payload.get("timestamps")
        or payload.get("words")
    )
    if not isinstance(raw_segments, list) or not raw_segments:
        return []

    segments = []
    for item in raw_segments:
        text = str(item.get("text") or item.get("word") or "").strip()
        if not text:
            continue
        start = item.get("start", item.get("start_time", 0))
        end = item.get("end", item.get("end_time", start))
        segments.append(
            TranscriptSegment(
                id=str(item.get("id") or uuid.uuid4().hex),
                start=round(float(start or 0) + offset, 2),
                end=round(float(end or start or 0) + offset, 2),
                text=text,
                language=str(item.get("language") or language),
            )
        )
    return _merge_transcript_segments(segments)


def _segments_from_payload(payload: dict, offset: float = 0, duration: float | None = None) -> list[TranscriptSegment]:
    language = str(payload.get("language") or payload.get("detected_language") or "zh")
    segments = _timestamp_segments_from_payload(payload, language, offset=offset)
    if segments:
        return segments

    text = _clean_asr_text(str(payload.get("text") or payload.get("transcript") or ""))
    return segments_from_text(text, language=language, offset=offset, duration=duration)


def _clean_asr_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^language\s+\w+\s*", "", text, flags=re.I)
    text = re.sub(r"</?asr_text>", "", text, flags=re.I)
    return text.strip()


def _clean_llm_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


async def _complete_llm(messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
    settings = _settings()
    llm_base_url = str(settings.get("llm_base_url") or LLM_BASE_URL).rstrip("/")
    if not llm_base_url:
        raise RuntimeError("未配置大模型服务地址，无法生成智能纪要。")

    payload = {
        "model": str(settings.get("llm_model") or LLM_MODEL),
        "messages": messages,
        "temperature": float(settings.get("llm_temperature", 0.2)),
        "max_tokens": int(max_tokens or settings.get("llm_max_tokens", 1200)),
    }
    headers = {"Authorization": f"Bearer {settings.get('llm_api_key') or LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{llm_base_url}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"LLM 服务返回 {response.status_code}: {detail}")
        data = response.json()
    return _clean_llm_text(str(data["choices"][0]["message"]["content"]))


def _json_object_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects[-1] if objects else {}


def _jsonish_stage_from_text(text: str) -> dict:
    def array_value(key: str) -> list[str]:
        values: list[list[str]] = []
        for match in re.finditer(rf'"{key}"\s*:\s*\[(.*?)\]', text, flags=re.S):
            items = [
                item.strip()
                for item in re.findall(r'"([^"]+)"', match.group(1), flags=re.S)
                if item.strip()
            ]
            if items:
                values.append(items)
        placeholders = {"摘要短句", "结论短句", "待办短句"}
        for items in reversed(values):
            if not all(item in placeholders for item in items):
                return items
        return values[-1] if values else []

    title_matches = [item.strip() for item in re.findall(r'"title"\s*:\s*"([^"]+)"', text, flags=re.S) if item.strip()]
    title = next((item for item in reversed(title_matches) if item != "本阶段主题"), title_matches[-1] if title_matches else "")
    summary = array_value("summary")
    conclusions = array_value("conclusions")
    todos = array_value("todos")
    if not title and not summary and not conclusions and not todos:
        return {}
    if not any(_stage_line_valid(item) for item in [title, *summary, *conclusions, *todos]):
        return {}
    return {
        "title": title or "阶段摘要",
        "summary": summary,
        "conclusions": conclusions,
        "todos": todos,
    }


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in re.split(r"\n+|(?<=[。！？；])\s*", value) if item.strip()]
    return []


def _stage_line_valid(text: str) -> bool:
    text = text.strip()
    if not text or text == "暂无":
        return False
    return text not in {"本阶段主题", "摘要短句", "结论短句", "待办短句", "阶段摘要"}


def _stage_list_from_value(value: object, fallback: list[str] | None = None) -> list[str]:
    items = [item for item in _as_list(value) if _stage_line_valid(item)]
    return items or (fallback or [])


def _clean_markdown_item(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*[-*]\s+", "", text)
    text = re.sub(r"^\s*\d+[.、]\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _minutes_sections(document: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in document.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            current = re.sub(r"^#+\s*", "", line)
            current = re.sub(r"^[一二三四五六七八九十]+[、.]\s*", "", current).strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("#"):
            continue
        if not current:
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _section_items(sections: dict[str, list[str]], keyword: str) -> list[str]:
    lines: list[str] = []
    for title, items in sections.items():
        if keyword not in title:
            continue
        for item in items:
            if item.startswith("|") or re.match(r"^\|?\s*:?-{2,}", item):
                continue
            clean = _clean_markdown_item(item)
            if clean and not clean.startswith(("日期", "参会人员", "纪要整理时间", "记录人")):
                lines.append(clean)
    return lines


def _todo_items_from_minutes(sections: dict[str, list[str]]) -> list[SummaryTodo]:
    todos: list[SummaryTodo] = []
    for title, lines in sections.items():
        if "待办事项" not in title:
            continue
        for line in lines:
            if not line.startswith("|") or "---" in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 4 or cells[0] == "序号":
                continue
            owner = cells[1] or "待补充"
            task = _clean_markdown_item(cells[2])
            due = cells[3] or "待补充"
            if due in {"高", "中", "低"}:
                task = f"{task}（{due}优先级）"
                due = "待补充"
            if _valid_card_text(task):
                todos.append(SummaryTodo(owner=owner, task=task, due=due))
    for item in _section_items(sections, "下一步行动"):
        if _valid_card_text(item):
            todos.append(SummaryTodo(owner="待补充", task=item, due="待补充"))
    return todos


def _valid_card_text(text: str) -> bool:
    if not text.strip():
        return False
    placeholders = {"一句话总览", "结论内容", "待办内容", "问题内容", "详情标题", "详情短句"}
    return text.strip() not in placeholders


def _summary_cards_from_dict(data: dict, fallback_text: str = "") -> SummaryCards:
    overview = str(data.get("overview") or "").strip()
    if not _valid_card_text(overview):
        overview = ""

    conclusions: list[SummaryConclusion] = []
    for item in data.get("conclusions") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            status = str(item.get("status") or "已确定").strip() or "已确定"
        else:
            text = str(item).strip()
            status = "已确定"
        if _valid_card_text(text):
            conclusions.append(SummaryConclusion(status=status, text=text))

    todos: list[SummaryTodo] = []
    for item in data.get("todos") or []:
        if isinstance(item, dict):
            task = str(item.get("task") or item.get("text") or "").strip()
            owner = str(item.get("owner") or "待补充").strip() or "待补充"
            due = str(item.get("due") or "待补充").strip() or "待补充"
        else:
            task = str(item).strip()
            owner = "待补充"
            due = "待补充"
        if _valid_card_text(task) and task != "暂无":
            todos.append(SummaryTodo(owner=owner, task=task, due=due))

    risks: list[SummaryRisk] = []
    for item in data.get("risks") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            level = str(item.get("level") or "中").strip() or "中"
        else:
            text = str(item).strip()
            level = "中"
        if _valid_card_text(text) and text != "暂无":
            risks.append(SummaryRisk(level=level, text=text))

    details: list[SummaryDetailGroup] = []
    for item in data.get("details") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        items = [line for line in _as_list(item.get("items")) if _valid_card_text(line)]
        if title and items:
            details.append(SummaryDetailGroup(title=title, items=items))

    fallback_items = _split_transcript_text(fallback_text)[:5] if fallback_text else []
    if not overview and fallback_items:
        overview = fallback_items[0]
    if not details and fallback_items:
        details = [SummaryDetailGroup(title="讨论摘要", items=fallback_items)]

    return SummaryCards(
        overview=overview,
        conclusions=conclusions,
        todos=todos,
        risks=risks,
        details=details,
    )


def _summary_cards_from_minutes_document(document: str) -> SummaryCards:
    sections = _minutes_sections(document)
    if not sections:
        return SummaryCards()

    core_items = _section_items(sections, "会议核心议题")
    discussion_items = _section_items(sections, "主要讨论内容")
    risk_items = _section_items(sections, "关键问题")
    conclusion_items = _section_items(sections, "关键结论")

    return SummaryCards(
        overview=core_items[0] if core_items else "",
        conclusions=[
            SummaryConclusion(status="已确定", text=item)
            for item in conclusion_items
            if _valid_card_text(item)
        ],
        todos=_todo_items_from_minutes(sections),
        risks=[
            SummaryRisk(level="中", text=item)
            for item in risk_items
            if _valid_card_text(item) and item != "暂无"
        ],
        details=[
            SummaryDetailGroup(title="主要讨论内容", items=discussion_items)
        ]
        if discussion_items
        else [],
    )


def _merge_summary_cards(primary: SummaryCards, fallback: SummaryCards) -> SummaryCards:
    return SummaryCards(
        overview=primary.overview or fallback.overview,
        conclusions=primary.conclusions or fallback.conclusions,
        todos=primary.todos or fallback.todos,
        risks=primary.risks or fallback.risks,
        details=primary.details or fallback.details,
    )


def _asr_error(response: httpx.Response) -> RuntimeError:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message") or payload.get("detail") or response.text
    except Exception:
        message = response.text
    return RuntimeError(f"ASR 服务返回 {response.status_code}: {message}")


def _convert_to_wav(file_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法把浏览器录音转换为 ASR 支持的 wav。")

    target = file_path.with_name(f"{file_path.stem}-asr.wav")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(file_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"音频转换失败: {(result.stderr or result.stdout).strip()}")
    return target


async def transcribe_audio(file_path: Path, offset: float = 0) -> list[TranscriptSegment]:
    settings = _settings()
    url = _require_asr(settings)
    headers = {"Authorization": f"Bearer {settings.get('asr_api_key') or ASR_API_KEY}"}
    model = str(settings.get("asr_model") or ASR_MODEL)
    max_retries = int(settings.get("asr_max_retries") if settings.get("asr_max_retries") is not None else ASR_MAX_RETRIES)
    last_error: Exception | None = None
    asr_file = _convert_to_wav(file_path)
    duration = _duration_seconds(asr_file)

    for _attempt in range(max_retries + 1):
        try:
            with asr_file.open("rb") as audio:
                files = {"file": (asr_file.name, audio, "audio/wav")}
                data = {
                    "model": model,
                    "response_format": "json",
                }
                async with httpx.AsyncClient(timeout=600) as client:
                    response = await client.post(url, headers=headers, files=files, data=data)
                    if response.status_code >= 400:
                        raise _asr_error(response)
                    payload = response.json()
            return _segments_from_payload(payload, offset=offset, duration=duration)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"ASR 转写失败，已重试 {max_retries} 次：{last_error}") from last_error


async def summarize_meeting(title: str, transcript_text: str) -> str:
    if not transcript_text.strip():
        raise RuntimeError("当前会议还没有转写文字，无法生成智能纪要。")

    return await _complete_llm(
        [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "你是会议纪要助手。基于转写原文生成简洁、准确、便于阅读的中文会议纪要。"
                    "只使用中文输出，不要英文标题，不要代码块，不要表格，不要寒暄，不要输出思考过程。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n\n"
                    "请严格按下面栏目输出，每个栏目用短句列表，不要写成一大段：\n"
                    "会议摘要：\n"
                    "- 用三到五条概括会议讨论内容。\n\n"
                    "关键结论：\n"
                    "- 列出已经形成的决定、共识或判断。\n\n"
                    "待跟进事项：\n"
                    "- 列出负责人、事项和时间；没有明确负责人或时间也要如实说明。\n\n"
                    "风险与问题：\n"
                    "- 列出仍未解决的问题、风险或依赖。\n\n"
                    "没有明确内容的栏目只写“- 暂无”。\n\n"
                    f"转写原文：\n{transcript_text}"
                ),
            },
        ]
    )


def summary_cards_to_text(cards: SummaryCards) -> str:
    lines = ["会议摘要："]
    lines.append(f"- {cards.overview or '暂无'}")
    if cards.details:
        for group in cards.details:
            lines.append(f"- {group.title}：{'；'.join(group.items[:3])}")

    lines.append("\n关键结论：")
    if cards.conclusions:
        lines.extend(f"- {item.status}：{item.text}" for item in cards.conclusions)
    else:
        lines.append("- 暂无")

    lines.append("\n待跟进事项：")
    if cards.todos:
        lines.extend(f"- {item.owner}：{item.task}（{item.due}）" for item in cards.todos)
    else:
        lines.append("- 暂无")

    lines.append("\n风险与问题：")
    if cards.risks:
        lines.extend(f"- {item.level}：{item.text}" for item in cards.risks)
    else:
        lines.append("- 暂无")
    return "\n".join(lines)


async def summarize_meeting_cards(title: str, transcript_text: str, minutes_document: str = "") -> SummaryCards:
    if not transcript_text.strip():
        raise RuntimeError("当前会议还没有转写文字，无法生成智能纪要。")

    source_text = minutes_document.strip() or transcript_text.strip()
    source_name = "正式会议纪要" if minutes_document.strip() else "转写原文"
    content = await _complete_llm(
        [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "你是会议纪要产品的信息架构助手。请把整理后的会议内容提炼成适合卡片界面展示的结构化中文纪要。"
                    "只输出一个 JSON 对象，不要代码块，不要表格，不要英文内容，不要思考过程。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n\n"
                    "请优先依据正式会议纪要提炼，不要直接复述寒暄、口误、重复词和无效口语。"
                    "如果来源是转写原文，也要先理解真实议题，再输出经过整理的纪要卡片。\n\n"
                    "请严格输出 JSON，字段如下：\n"
                    "{\n"
                    "  \"overview\": \"概括会议核心议题和目标，35字以内\",\n"
                    "  \"conclusions\": [{\"status\": \"已确定/待确认/方向一致\", \"text\": \"从关键结论与共识中提炼的结论\"}],\n"
                    "  \"todos\": [{\"owner\": \"负责人，未知写待补充\", \"task\": \"从待办事项和下一步行动中提炼的任务\", \"due\": \"截止时间，未知写待补充\"}],\n"
                    "  \"risks\": [{\"level\": \"高/中/低\", \"text\": \"从关键问题与争议中提炼的问题或风险\"}],\n"
                    "  \"details\": [{\"title\": \"讨论模块标题\", \"items\": [\"从主要讨论内容中提炼的要点\"]}]\n"
                    "}\n"
                    "要求：\n"
                    "1. overview 写会议真正讨论的事情，不要写第一句转写原文。\n"
                    "2. details 按议题分组，每组二到四条，优先覆盖正式纪要里的“主要讨论内容”。\n"
                    "3. conclusions 对应“关键结论与共识”，todos 对应“待办事项”和“下一步行动”，risks 对应“关键问题与争议”。\n"
                    "4. 没有明确内容用空数组，不要编造负责人、时间或已经达成的结论。\n"
                    "5. 所有内容必须是中文短句，适合卡片阅读。\n\n"
                    f"{source_name}：\n{source_text}"
                ),
            },
        ],
        max_tokens=2200,
    )
    data = _json_object_from_text(content)
    cards = _summary_cards_from_dict(data)
    if minutes_document.strip():
        cards = _merge_summary_cards(cards, _summary_cards_from_minutes_document(minutes_document))
    if not cards.overview and not cards.details and not cards.conclusions:
        cards = _summary_cards_from_dict({}, fallback_text=source_text)
    return cards


async def summarize_stage(title: str, start: float, end: float, transcript_text: str) -> dict:
    if not transcript_text.strip():
        raise RuntimeError("当前时间段还没有转写文字，无法生成阶段摘要。")

    content = await _complete_llm(
        [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "你是会议实时纪要助手。请把两分钟左右的转写片段整理成中文阶段摘要。"
                    "只输出一个 JSON 对象，不要代码块，不要英文键以外的英文内容，不要思考过程。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n"
                    f"时间段：{start:.0f}秒到{end:.0f}秒\n\n"
                    "请输出 JSON，格式严格如下：\n"
                    "{\"title\":\"本阶段主题\",\"summary\":[\"摘要短句\"],\"conclusions\":[\"结论短句\"],\"todos\":[\"待办短句\"]}\n"
                    "要求：summary 二到四条；conclusions 没有就用 [\"暂无\"]；todos 没有就用 [\"暂无\"]。\n\n"
                    f"转写原文：\n{transcript_text}"
                ),
            },
        ],
        max_tokens=800,
    )
    data = _json_object_from_text(content)
    if not data:
        data = _jsonish_stage_from_text(content)
    fallback_summary = _split_transcript_text(transcript_text)[:4] or [transcript_text.strip()]
    if not data and re.search(r"thinking process|analyze user input|mental refinement|思考过程", content, flags=re.I):
        return {
            "title": "阶段摘要",
            "summary": fallback_summary,
            "conclusions": ["暂无"],
            "todos": ["暂无"],
        }
    title_text = str(data.get("title") or "阶段摘要").strip() or "阶段摘要"
    if not _stage_line_valid(title_text):
        title_text = f"{start:.0f}-{end:.0f}秒阶段摘要"
    return {
        "title": title_text,
        "summary": _stage_list_from_value(data.get("summary"), fallback_summary),
        "conclusions": _stage_list_from_value(data.get("conclusions"), ["暂无"]),
        "todos": _stage_list_from_value(data.get("todos"), ["暂无"]),
    }


async def write_minutes_document(title: str, transcript_text: str, stage_summary_text: str = "") -> str:
    if not transcript_text.strip():
        raise RuntimeError("当前会议还没有转写文字，无法生成正式纪要。")

    stage_summary_text = stage_summary_text.strip()
    transcript_text = transcript_text.strip()
    has_stage_summary = bool(stage_summary_text)
    primary_source_label = "阶段摘要合集" if has_stage_summary else "完整转写原文"
    secondary_source_label = "完整转写原文" if has_stage_summary else "暂无补充"
    return await _complete_llm(
        [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "你是专业中文会议纪要整理助手。请生成正式、清楚、可交付的 Markdown 会议纪要。"
                    "如果提供了阶段摘要合集，请以它作为主要依据，完整转写原文只作为补充核对，不要直接照搬全文。"
                    "不要输出思考过程，不要英文标题，不要代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n\n"
                    "请参考示例风格生成正式纪要，必须包含这些中文章节：\n"
                    "# 会议纪要：会议主题\n"
                    "日期、参会人员如果原文没有明确就写待补充。\n"
                    "## 一、会议核心议题\n"
                    "## 二、主要讨论内容\n"
                    "## 三、关键问题与争议\n"
                    "## 四、待办事项\n"
                    "待办事项用 Markdown 表格，列为：序号、负责人、事项、优先级。\n"
                    "## 五、关键结论与共识\n"
                    "## 六、下一步行动\n"
                    "末尾写“纪要整理时间”和“记录人：苏小智”。\n\n"
                    f"{primary_source_label}：\n{stage_summary_text or transcript_text}\n\n"
                    f"{secondary_source_label}：\n{transcript_text if has_stage_summary else '暂无'}"
                ),
            },
        ],
        max_tokens=4000,
    )


async def stream_start() -> str:
    stream_base_url = str(_settings().get("asr_stream_base_url") or ASR_STREAM_BASE_URL).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{stream_base_url}/api/start")
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 启动失败 {response.status_code}: {response.text}")
            return str(response.json()["session_id"])
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 启动超时，可能有旧录音会话正在占用模型，请稍后重试。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc


async def stream_chunk(session_id: str, pcm: bytes) -> dict[str, str]:
    stream_base_url = str(_settings().get("asr_stream_base_url") or ASR_STREAM_BASE_URL).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{stream_base_url}/api/chunk",
                params={"session_id": session_id},
                content=pcm,
                headers={"Content-Type": "application/octet-stream"},
            )
            if response.status_code == 409:
                return {"language": "", "text": ""}
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 识别失败 {response.status_code}: {response.text}")
            payload = response.json()
            return {"language": str(payload.get("language") or ""), "text": str(payload.get("text") or "")}
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 识别超时。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc


async def stream_finish(session_id: str) -> dict[str, str]:
    stream_base_url = str(_settings().get("asr_stream_base_url") or ASR_STREAM_BASE_URL).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(f"{stream_base_url}/api/finish", params={"session_id": session_id})
            if response.status_code == 409:
                return {"language": "", "text": ""}
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 收尾失败 {response.status_code}: {response.text}")
            payload = response.json()
            return {"language": str(payload.get("language") or ""), "text": str(payload.get("text") or "")}
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 收尾超时，已释放本地录音状态。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc
