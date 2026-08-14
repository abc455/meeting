from pydantic import BaseModel, Field


class MeetingCreate(BaseModel):
    title: str = Field(min_length=1)


class Meeting(MeetingCreate):
    id: str
    status: str
    created_at: str


class TranscriptSegment(BaseModel):
    id: str
    start: float = 0
    end: float = 0
    text: str
    language: str = "zh"


class StageSummary(BaseModel):
    id: str
    start: float = 0
    end: float = 0
    title: str = "阶段摘要"
    summary: list[str] = []
    conclusions: list[str] = []
    todos: list[str] = []
    created_at: str = ""


class SummaryConclusion(BaseModel):
    status: str = "已确定"
    text: str


class SummaryTodo(BaseModel):
    owner: str = "待补充"
    task: str
    due: str = "待补充"


class SummaryRisk(BaseModel):
    level: str = "中"
    text: str


class SummaryDetailGroup(BaseModel):
    title: str
    items: list[str] = []


class SummaryCards(BaseModel):
    overview: str = ""
    conclusions: list[SummaryConclusion] = []
    todos: list[SummaryTodo] = []
    risks: list[SummaryRisk] = []
    details: list[SummaryDetailGroup] = []


class MeetingDetail(BaseModel):
    meeting: Meeting
    transcript: list[TranscriptSegment] = []
    summary: str = ""
    summary_cards: SummaryCards = Field(default_factory=SummaryCards)
    stage_summaries: list[StageSummary] = []
    minutes_document: str = ""


class AudioTranscriptionResult(BaseModel):
    text: str
    segments: list[TranscriptSegment] = []


class SummaryResult(BaseModel):
    summary: str
    summary_cards: SummaryCards = Field(default_factory=SummaryCards)
    minutes_document: str = ""


class StageSummaryResult(BaseModel):
    stage_summaries: list[StageSummary] = []


class MinutesDocumentResult(BaseModel):
    document: str


class AppSettings(BaseModel):
    asr_base_url: str = ""
    asr_model: str = "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B"
    asr_api_key: str = "EMPTY"
    asr_transcribe_path: str = "/v1/audio/transcriptions"
    asr_max_retries: int = Field(default=2, ge=0, le=10)
    asr_stream_base_url: str = "http://127.0.0.1:8006"
    llm_base_url: str = ""
    llm_model: str = "Qwen3.6-35B-A3B-NVFP4"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=1200, ge=128, le=16000)
