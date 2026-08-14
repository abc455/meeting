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


class MeetingDetail(BaseModel):
    meeting: Meeting
    transcript: list[TranscriptSegment] = []
    summary: str = ""


class AudioTranscriptionResult(BaseModel):
    text: str
    segments: list[TranscriptSegment] = []


class SummaryResult(BaseModel):
    summary: str
