from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .schemas import TranscriptSegment


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ASR_BASE_URL = os.getenv("ASR_BASE_URL", "").rstrip("/")
ASR_API_KEY = os.getenv("ASR_API_KEY", "EMPTY")
ASR_MODEL = os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B")
ASR_TRANSCRIBE_PATH = os.getenv("ASR_TRANSCRIBE_PATH", "/v1/audio/transcriptions")
ASR_MAX_RETRIES = int(os.getenv("ASR_MAX_RETRIES", "2"))
ASR_STREAM_BASE_URL = os.getenv("ASR_STREAM_BASE_URL", "http://127.0.0.1:8005").rstrip("/")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-NVFP4")


def _require_asr() -> str:
    if not ASR_BASE_URL:
        raise RuntimeError("未配置 ASR_BASE_URL，无法进行真实转写。")
    return f"{ASR_BASE_URL}{ASR_TRANSCRIBE_PATH}"


def _segments_from_payload(payload: dict, offset: float = 0) -> list[TranscriptSegment]:
    language = str(payload.get("language") or payload.get("detected_language") or "zh")
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments = []
        for item in raw_segments:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    id=str(item.get("id") or uuid.uuid4().hex),
                    start=round(float(item.get("start", 0)) + offset, 2),
                    end=round(float(item.get("end", 0)) + offset, 2),
                    text=text,
                    language=str(item.get("language") or language),
                )
            )
        return segments

    text = _clean_asr_text(str(payload.get("text") or payload.get("transcript") or ""))
    if not text:
        return []
    return [
        TranscriptSegment(
            id=uuid.uuid4().hex,
            start=round(offset, 2),
            end=round(offset, 2),
            text=text,
            language=language,
        )
    ]


def _clean_asr_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^language\s+\w+\s*", "", text, flags=re.I)
    text = re.sub(r"</?asr_text>", "", text, flags=re.I)
    return text.strip()


def _clean_llm_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


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
    url = _require_asr()
    headers = {"Authorization": f"Bearer {ASR_API_KEY}"}
    last_error: Exception | None = None
    asr_file = _convert_to_wav(file_path)

    for _attempt in range(ASR_MAX_RETRIES + 1):
        try:
            with asr_file.open("rb") as audio:
                files = {"file": (asr_file.name, audio, "audio/wav")}
                data = {
                    "model": ASR_MODEL,
                    "response_format": "json",
                }
                async with httpx.AsyncClient(timeout=600) as client:
                    response = await client.post(url, headers=headers, files=files, data=data)
                    if response.status_code >= 400:
                        raise _asr_error(response)
                    payload = response.json()
            return _segments_from_payload(payload, offset=offset)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"ASR 转写失败，已重试 {ASR_MAX_RETRIES} 次：{last_error}") from last_error


async def summarize_meeting(title: str, transcript_text: str) -> str:
    if not LLM_BASE_URL:
        raise RuntimeError("未配置 LLM_BASE_URL，无法生成真实 AI 纪要。")
    if not transcript_text.strip():
        raise RuntimeError("当前会议还没有转写文字，无法生成 AI 纪要。")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是会议纪要助手。基于转写原文生成简洁中文会议纪要，只输出正文，不要 Markdown 代码块。",
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n\n"
                    "请输出：1. 会议摘要；2. 关键结论；3. 待跟进事项。"
                    "没有明确内容的部分写“暂无”。\n\n"
                    f"转写原文：\n{transcript_text}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"LLM 服务返回 {response.status_code}: {detail}")
        data = response.json()
    return _clean_llm_text(str(data["choices"][0]["message"]["content"]))


async def stream_start() -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{ASR_STREAM_BASE_URL}/api/start")
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 启动失败 {response.status_code}: {response.text}")
            return str(response.json()["session_id"])
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 启动超时，可能有旧录音会话正在占用模型，请稍后重试。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc


async def stream_chunk(session_id: str, pcm: bytes) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{ASR_STREAM_BASE_URL}/api/chunk",
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
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(f"{ASR_STREAM_BASE_URL}/api/finish", params={"session_id": session_id})
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
