import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Check, Clipboard, Clock3, Database, Download, FileAudio, FileText, FlaskConical, HardDrive, Home, Loader2, Mic, PanelLeftClose, PanelLeftOpen, Plus, Search, Settings, SlidersHorizontal, Sparkles, Square, Trash2 } from "lucide-react";
import "./styles.css";

type Meeting = {
  id: string;
  title: string;
  status: string;
  created_at: string;
};

type Segment = {
  id: string;
  start: number;
  end: number;
  text: string;
  language: string;
};

type MeetingDetail = {
  meeting: Meeting;
  transcript: Segment[];
  summary: string;
  summary_cards: SummaryCards;
  stage_summaries: StageSummary[];
  minutes_document: string;
};

type TranscriptionResult = {
  text: string;
  segments: Segment[];
};

type SummaryResult = {
  summary: string;
  summary_cards: SummaryCards;
  minutes_document: string;
};

type StageSummary = {
  id: string;
  start: number;
  end: number;
  title: string;
  summary: string[];
  conclusions: string[];
  todos: string[];
  created_at: string;
};

type StageSummaryResult = {
  stage_summaries: StageSummary[];
};

type MinutesDocumentResult = {
  document: string;
};

type StreamStartResult = {
  session_id: string;
};

type SummarySection = {
  title: string;
  items: string[];
};

type SummaryConclusion = {
  status: string;
  text: string;
};

type SummaryTodo = {
  owner: string;
  task: string;
  due: string;
};

type SummaryRisk = {
  level: string;
  text: string;
};

type SummaryDetailGroup = {
  title: string;
  items: string[];
};

type SummaryCards = {
  overview: string;
  conclusions: SummaryConclusion[];
  todos: SummaryTodo[];
  risks: SummaryRisk[];
  details: SummaryDetailGroup[];
};

type TodoInteraction = {
  done?: boolean;
  removed?: boolean;
};

type AppSettings = {
  asr_base_url: string;
  asr_model: string;
  asr_api_key: string;
  asr_transcribe_path: string;
  asr_max_retries: number;
  asr_stream_base_url: string;
  llm_base_url: string;
  llm_model: string;
  llm_api_key: string;
  llm_temperature: number;
  llm_max_tokens: number;
};


const API_BASE = "/api";
const DEFAULT_SETTINGS: AppSettings = {
  asr_base_url: "",
  asr_model: "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B",
  asr_api_key: "EMPTY",
  asr_transcribe_path: "/v1/audio/transcriptions",
  asr_max_retries: 2,
  asr_stream_base_url: "http://127.0.0.1:8005",
  llm_base_url: "",
  llm_model: "Qwen3.6-35B-A3B-NVFP4",
  llm_api_key: "EMPTY",
  llm_temperature: 0.2,
  llm_max_tokens: 1200,
};

const EMPTY_SUMMARY_CARDS: SummaryCards = {
  overview: "",
  conclusions: [],
  todos: [],
  risks: [],
  details: [],
};

const TARGET_SAMPLE_RATE = 16000;
const STREAM_CHUNK_SAMPLES = TARGET_SAMPLE_RATE * 0.5;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const raw = await response.text();
    let message = raw.trim();
    if (raw) {
      try {
        const payload = JSON.parse(raw) as { detail?: unknown; error?: { message?: unknown } };
        const detail = payload.detail;
        const errorMessage = payload.error?.message;
        if (typeof detail === "string" && detail.trim()) message = detail.trim();
        else if (typeof errorMessage === "string" && errorMessage.trim()) message = errorMessage.trim();
        else if (detail) message = JSON.stringify(detail);
      } catch {
        message = raw.trim();
      }
    }
    throw new Error(message || `请求失败：HTTP ${response.status} ${response.statusText || ""}`.trim());
  }
  return response.json() as Promise<T>;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  let timeoutId = 0;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = window.setTimeout(() => reject(new Error(message)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function formatSegmentTime(value: number) {
  const total = Math.max(0, Math.floor(value || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return [hours, minutes, seconds].map((item) => String(item).padStart(2, "0")).join(":");
}

function formatTimeRange(start: number, end: number) {
  return `${formatSegmentTime(start)} - ${formatSegmentTime(end)}`;
}

function safeFileName(value: string) {
  return value.replace(/[\\/:*?"<>|]+/g, "-").trim() || "会议纪要";
}

function transcriptText(segments: Segment[]) {
  return segments.map((item) => item.text).filter(Boolean).join("\n");
}

function cleanSummaryLine(line: string) {
  return line
    .replace(/^#{1,6}\s*/, "")
    .replace(/^[\s>*-]+/, "")
    .replace(/^\d+[.)、]\s*/, "")
    .replace(/^\*+|\*+$/g, "")
    .trim();
}

function splitSummaryItems(text: string) {
  const normalized = text.replace(/\r/g, "").trim();
  if (!normalized) return [];
  const lines = normalized
    .split(/\n+/)
    .map(cleanSummaryLine)
    .filter(Boolean);
  if (lines.length > 1) return lines;
  return normalized
    .split(/(?<=[。！？；])\s*/)
    .map(cleanSummaryLine)
    .filter(Boolean);
}

function parseSummarySections(text: string): SummarySection[] {
  const titles = ["会议摘要", "关键结论", "待跟进事项", "风险与问题"];
  const sections = titles.map((title) => ({ title, items: [] as string[] }));
  const byTitle = new Map(sections.map((section) => [section.title, section]));
  let current = byTitle.get("会议摘要")!;
  let sawHeading = false;

  const lines = text
    .replace(/\r/g, "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) return sections;

  for (const rawLine of lines) {
    const line = cleanSummaryLine(rawLine);
    const heading = titles.find((title) => new RegExp(`^${title}\\s*[:：]?$`).test(line));
    const inlineHeading = titles.find((title) => line.startsWith(`${title}：`) || line.startsWith(`${title}:`));
    if (heading) {
      sawHeading = true;
      current = byTitle.get(heading)!;
      continue;
    }
    if (inlineHeading) {
      sawHeading = true;
      current = byTitle.get(inlineHeading)!;
      const content = line.slice(inlineHeading.length).replace(/^[:：]\s*/, "").trim();
      if (content) current.items.push(...splitSummaryItems(content));
      continue;
    }
    current.items.push(line);
  }

  if (!sawHeading) {
    sections.forEach((section) => {
      section.items = [];
    });
    byTitle.get("会议摘要")!.items = splitSummaryItems(text);
  }
  return sections;
}

function isEmptySummaryCards(cards?: SummaryCards) {
  if (!cards) return true;
  return !cards.overview.trim() && cards.conclusions.length === 0 && cards.todos.length === 0 && cards.risks.length === 0 && cards.details.length === 0;
}

function summaryCardsFromSections(sections: SummarySection[]): SummaryCards {
  const getItems = (title: string) => sections.find((section) => section.title === title)?.items.filter((item) => item && item !== "暂无") || [];
  const summaryItems = getItems("会议摘要");
  const conclusionItems = getItems("关键结论");
  const todoItems = getItems("待跟进事项");
  const riskItems = getItems("风险与问题");
  return {
    overview: summaryItems[0] || "",
    conclusions: conclusionItems.map((text) => ({ status: text.includes("待") ? "待确认" : "已确定", text })),
    todos: todoItems.map((task) => ({ owner: "待补充", task, due: "待补充" })),
    risks: riskItems.map((text) => ({ level: "中", text })),
    details: summaryItems.length > 1 ? [{ title: "详细摘要", items: summaryItems.slice(1) }] : [],
  };
}

function canUseMicrophone() {
  const mediaDevices = (navigator as Navigator & { mediaDevices?: MediaDevices }).mediaDevices;
  return Boolean(mediaDevices && typeof mediaDevices.getUserMedia === "function" && window.isSecureContext);
}

function downsample(input: Float32Array, sourceRate: number, targetRate: number) {
  if (sourceRate === targetRate) return input;
  const ratio = sourceRate / targetRate;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(Math.floor((index + 1) * ratio), input.length);
    let sum = 0;
    for (let cursor = start; cursor < end; cursor += 1) sum += input[cursor];
    output[index] = sum / Math.max(1, end - start);
  }
  return output;
}

function App() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [meetingSearch, setMeetingSearch] = useState("");
  const [uploadText, setUploadText] = useState("");
  const [uploadSegments, setUploadSegments] = useState<Segment[]>([]);
  const [streamText, setStreamText] = useState("");
  const [streamSegments, setStreamSegments] = useState<Segment[]>([]);
  const [summaryText, setSummaryText] = useState("");
  const [summaryCards, setSummaryCards] = useState<SummaryCards>(EMPTY_SUMMARY_CARDS);
  const [stageSummaries, setStageSummaries] = useState<StageSummary[]>([]);
  const [minutesDocument, setMinutesDocument] = useState("");
  const [activeView, setActiveView] = useState<"transcript" | "summary">("transcript");
  const [summaryTab, setSummaryTab] = useState<"timeline" | "cards" | "document">("timeline");
  const [todoInteractions, setTodoInteractions] = useState<Record<string, TodoInteraction>>({});
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isStartingRecording, setIsStartingRecording] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"general" | "recordings" | "transcript" | "summary" | "beta">("general");
  const [settingsDraft, setSettingsDraft] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [settingsMessage, setSettingsMessage] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const microphoneReady = canUseMicrophone();

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const monitorGainRef = useRef<GainNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const bufferedSamplesRef = useRef<Float32Array[]>([]);
  const pendingChunksRef = useRef<Float32Array[]>([]);
  const isPushingRef = useRef(false);
  const streamSessionIdRef = useRef("");
  const streamMeetingIdRef = useRef("");
  const selectedIdRef = useRef("");
  const recordingActiveRef = useRef(false);
  const stageSummaryRunningRef = useRef(false);
  const meetingSearchInputRef = useRef<HTMLInputElement | null>(null);
  const uploadFileInputRef = useRef<HTMLInputElement | null>(null);

  const currentTranscript = useMemo(() => transcriptText(detail?.transcript || []), [detail]);
  const filteredMeetings = useMemo(() => {
    const query = meetingSearch.trim().toLowerCase();
    if (!query) return meetings;
    return meetings.filter((meeting) => meeting.title.toLowerCase().includes(query) || meeting.status.toLowerCase().includes(query));
  }, [meetingSearch, meetings]);
  const transcriptRows = useMemo(() => {
    const storedSegments = detail?.transcript || [];
    const visibleSegments = streamSegments.length ? streamSegments : uploadSegments.length ? uploadSegments : storedSegments;
    if (visibleSegments.length) return visibleSegments;

    const fallbackText = streamText || uploadText;
    return fallbackText
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line, index) => ({
        id: `line-${index}`,
        start: index * 3,
        end: index * 3,
        text: line,
        language: "",
      }));
  }, [detail?.transcript, streamSegments, uploadSegments, streamText, uploadText]);
  const summarySections = useMemo(() => parseSummarySections(summaryText), [summaryText]);
  const displaySummaryCards = useMemo(() => isEmptySummaryCards(summaryCards) ? summaryCardsFromSections(summarySections) : summaryCards, [summaryCards, summarySections]);
  const hasSummaryContent = !isEmptySummaryCards(displaySummaryCards);
  const activeText = streamText || uploadText || currentTranscript;

  async function runBusy<T>(name: string, fn: () => Promise<T>) {
    setBusy(name);
    setError("");
    try {
      return await fn();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      setBusy("");
    }
  }

  async function loadMeetings(nextSelectedId?: string) {
    const data = await api<Meeting[]>("/meetings");
    setMeetings(data);
    const nextId = nextSelectedId !== undefined ? nextSelectedId : selectedId || data[0]?.id || "";
    setSelectedId(nextId);
    if (nextId) {
      await loadDetail(nextId);
    } else {
      setDetail(null);
      setUploadText("");
      setUploadSegments([]);
      setStreamText("");
      setStreamSegments([]);
      setSummaryText("");
      setSummaryCards(EMPTY_SUMMARY_CARDS);
      setStageSummaries([]);
      setMinutesDocument("");
    }
  }

  async function loadDetail(id = selectedId) {
    if (!id) return;
    const nextDetail = await api<MeetingDetail>(`/meetings/${id}`);
    setDetail(nextDetail);
    setSummaryText(nextDetail.summary || "");
    setSummaryCards(nextDetail.summary_cards || EMPTY_SUMMARY_CARDS);
    setStageSummaries(nextDetail.stage_summaries || []);
    setMinutesDocument(nextDetail.minutes_document || "");
  }

  async function createMeeting() {
    const nextTitle = meetingSearch.trim();
    if (!nextTitle) return;
    await runBusy("create", async () => {
      const meeting = await api<Meeting>("/meetings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: nextTitle }),
      });
      setMeetingSearch("");
      setUploadText("");
      setUploadSegments([]);
      setStreamText("");
      setStreamSegments([]);
      setSummaryText("");
      setSummaryCards(EMPTY_SUMMARY_CARDS);
      setStageSummaries([]);
      setMinutesDocument("");
      setActiveView("transcript");
      await loadMeetings(meeting.id);
    });
  }

  async function uploadAudio(file: File) {
    if (!detail) return;
    await runBusy("upload", async () => {
      const body = new FormData();
      body.append("file", file);
      const result = await api<TranscriptionResult>(`/meetings/${detail.meeting.id}/audio`, {
        method: "POST",
        body,
      });
      setUploadText(result.text);
      setUploadSegments(result.segments || []);
      setStreamText("");
      setStreamSegments([]);
      setSummaryText("");
      setSummaryCards(EMPTY_SUMMARY_CARDS);
      setStageSummaries([]);
      setMinutesDocument("");
      setActiveView("transcript");
      await loadDetail(detail.meeting.id);
    });
  }

  function mergeSamples(chunks: Float32Array[]) {
    const total = chunks.reduce((sum, item) => sum + item.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  function sampleCount(chunks: Float32Array[]) {
    return chunks.reduce((sum, item) => sum + item.length, 0);
  }

  function takeSamples(count: number) {
    const merged = mergeSamples(bufferedSamplesRef.current);
    const chunk = merged.slice(0, count);
    const rest = merged.slice(count);
    bufferedSamplesRef.current = rest.length ? [rest] : [];
    return chunk;
  }

  function queueReadyChunks(force = false) {
    if (!recordingActiveRef.current && !force) return;
    let total = sampleCount(bufferedSamplesRef.current);
    while (total >= STREAM_CHUNK_SAMPLES || (force && total > 0)) {
      const count = force && total < STREAM_CHUNK_SAMPLES ? total : STREAM_CHUNK_SAMPLES;
      pendingChunksRef.current.push(takeSamples(count));
      total = sampleCount(bufferedSamplesRef.current);
    }
    pumpStreamQueue();
  }

  async function postStreamChunk(id: string, sessionId: string, chunk: Float32Array) {
    const body = new ArrayBuffer(chunk.byteLength);
    new Float32Array(body).set(chunk);
    const result = await api<TranscriptionResult>(`/meetings/${id}/stream/chunk?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body,
    });
    if (streamSessionIdRef.current === sessionId) {
      setStreamText(result.text);
      setStreamSegments(result.segments || []);
    }
  }

  async function pumpStreamQueue() {
    const id = streamMeetingIdRef.current;
    const sessionId = streamSessionIdRef.current;
    if (!id || !sessionId || isPushingRef.current) return;

    isPushingRef.current = true;
    try {
      while (pendingChunksRef.current.length > 0 && streamSessionIdRef.current === sessionId) {
        const chunk = pendingChunksRef.current.shift();
        if (chunk?.length) await postStreamChunk(id, sessionId, chunk);
      }
    } catch (err) {
      pendingChunksRef.current = [];
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      isPushingRef.current = false;
      if (pendingChunksRef.current.length > 0 && streamSessionIdRef.current === sessionId) {
        pumpStreamQueue();
      }
    }
  }

  async function startRecording(meeting = detail?.meeting) {
    if (!meeting || isRecording || isStartingRecording) return;
    setError("");
    if (!microphoneReady) {
      setError("当前访问地址不能使用麦克风。请使用 HTTPS、localhost，或让公司 IT 通过浏览器策略把当前 HTTP 内网地址加入安全源。");
      return;
    }
    setUploadText("");
    setUploadSegments([]);
    setStreamText("");
    setStreamSegments([]);
    setActiveView("transcript");
    bufferedSamplesRef.current = [];
    pendingChunksRef.current = [];
    isPushingRef.current = false;
    setIsStartingRecording(true);
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const started = await api<StreamStartResult>(`/meetings/${meeting.id}/stream/start`, { method: "POST" });
      streamMeetingIdRef.current = meeting.id;
      streamSessionIdRef.current = started.session_id;
      recordingActiveRef.current = true;
      const AudioContextClass = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextClass) throw new Error("当前浏览器不支持 AudioContext。");
      const audioContext = new AudioContextClass();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      const monitorGain = audioContext.createGain();
      monitorGain.gain.value = 0;
      processor.onaudioprocess = (event) => {
        const pcm = downsample(new Float32Array(event.inputBuffer.getChannelData(0)), audioContext.sampleRate, TARGET_SAMPLE_RATE);
        bufferedSamplesRef.current.push(pcm);
        queueReadyChunks(false);
      };
      source.connect(processor);
      processor.connect(monitorGain);
      monitorGain.connect(audioContext.destination);

      mediaStreamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceRef.current = source;
      processorRef.current = processor;
      monitorGainRef.current = monitorGain;
      setIsRecording(true);
    } catch (err) {
      streamSessionIdRef.current = "";
      streamMeetingIdRef.current = "";
      recordingActiveRef.current = false;
      stream?.getTracks().forEach((track) => track.stop());
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsStartingRecording(false);
    }
  }

  async function startNewMeetingRecording() {
    if (isRecording || isStartingRecording || Boolean(busy)) return;
    const name = window.prompt("请输入会议名称");
    const nextTitle = name?.trim();
    if (!nextTitle) return;

    let meeting: Meeting | null = null;
    await runBusy("create", async () => {
      meeting = await api<Meeting>("/meetings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: nextTitle }),
      });
      setMeetingSearch("");
      setUploadText("");
      setUploadSegments([]);
      setStreamText("");
      setStreamSegments([]);
      setSummaryText("");
      setSummaryCards(EMPTY_SUMMARY_CARDS);
      setStageSummaries([]);
      setMinutesDocument("");
      setActiveView("transcript");
      setSettingsOpen(false);
      if (meeting) {
        setSelectedId(meeting.id);
        setDetail({ meeting, transcript: [], summary: "", summary_cards: EMPTY_SUMMARY_CARDS, stage_summaries: [], minutes_document: "" });
        await loadMeetings(meeting.id);
      }
    });
    if (meeting) await startRecording(meeting);
  }

  function stopRecording() {
    const meetingId = streamMeetingIdRef.current;
    const sessionId = streamSessionIdRef.current;
    recordingActiveRef.current = false;
    setIsRecording(false);
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    monitorGainRef.current?.disconnect();
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    queueReadyChunks(true);
    const finish = async () => {
      const deadline = Date.now() + 3000;
      while ((isPushingRef.current || pendingChunksRef.current.length > 0) && Date.now() < deadline) {
        await sleep(80);
        pumpStreamQueue();
      }
      pendingChunksRef.current = [];
      if (meetingId && sessionId) {
        const result = await withTimeout(
          api<TranscriptionResult>(`/meetings/${meetingId}/stream/finish?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" }),
          5000,
          "停止录音超时，已释放本地录音状态。"
        );
        setStreamText(result.text);
        setStreamSegments(result.segments || []);
      }
    };

    finish().catch((err) => setError(err instanceof Error ? err.message : String(err))).finally(() => {
      audioContextRef.current?.close();
      processorRef.current = null;
      sourceRef.current = null;
      monitorGainRef.current = null;
      audioContextRef.current = null;
      mediaStreamRef.current = null;
      streamSessionIdRef.current = "";
      streamMeetingIdRef.current = "";
      recordingActiveRef.current = false;
      bufferedSamplesRef.current = [];
      pendingChunksRef.current = [];
      loadDetail(meetingId || selectedIdRef.current).catch((err) => setError(String(err)));
    });
  }

  async function deleteMeeting(meeting: Meeting) {
    if (isRecording || isStartingRecording) return;
    const ok = window.confirm(`删除会议「${meeting.title}」？相关音频、转写和智能摘要也会一起删除。`);
    if (!ok) return;
    await runBusy(`delete-${meeting.id}`, async () => {
      await api<{ status: string }>(`/meetings/${meeting.id}`, { method: "DELETE" });
      const nextMeeting = meetings.find((item) => item.id !== meeting.id);
      const nextSelectedId = meeting.id === selectedId ? nextMeeting?.id || "" : selectedId;
      if (meeting.id === selectedId) {
        setDetail(null);
        setUploadText("");
        setUploadSegments([]);
        setStreamText("");
        setStreamSegments([]);
        setSummaryText("");
        setSummaryCards(EMPTY_SUMMARY_CARDS);
        setStageSummaries([]);
        setMinutesDocument("");
      }
      await loadMeetings(nextSelectedId);
    });
  }

  async function generateSummary() {
    if (!detail) return;
    await runBusy("summary", async () => {
      const result = await api<SummaryResult>(`/meetings/${detail.meeting.id}/summary`, { method: "POST" });
      setSummaryText(result.summary);
      setSummaryCards(result.summary_cards || EMPTY_SUMMARY_CARDS);
      setMinutesDocument(result.minutes_document || "");
      await loadDetail(detail.meeting.id);
    });
  }

  async function generateStageSummaries(silent = false) {
    if (!detail || stageSummaryRunningRef.current) return;
    stageSummaryRunningRef.current = true;
    const task = async () => {
      const result = await api<StageSummaryResult>(`/meetings/${detail.meeting.id}/stage-summaries${silent ? "" : "?refresh=true"}`, { method: "POST" });
      setStageSummaries(result.stage_summaries || []);
      await loadDetail(detail.meeting.id);
    };
    try {
      if (silent) {
        await task();
      } else {
        await runBusy("stage-summary", task);
      }
    } catch (err) {
      if (silent) setError(err instanceof Error ? err.message : String(err));
    } finally {
      stageSummaryRunningRef.current = false;
    }
  }

  async function generateMinutesDocument() {
    if (!detail) return;
    await runBusy("minutes-document", async () => {
      const result = await api<MinutesDocumentResult>(`/meetings/${detail.meeting.id}/minutes-document`, { method: "POST" });
      setMinutesDocument(result.document || "");
      await loadDetail(detail.meeting.id);
    });
  }

  function updateTodoInteraction(key: string, patch: TodoInteraction) {
    setTodoInteractions((current) => ({
      ...current,
      [key]: {
        ...current[key],
        ...patch,
      },
    }));
  }

  async function copyMinutesDocument() {
    if (!minutesDocument.trim()) return;
    await navigator.clipboard.writeText(minutesDocument);
  }

  function downloadMinutesDocument() {
    if (!detail || !minutesDocument.trim()) return;
    const blob = new Blob([minutesDocument], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeFileName(detail.meeting.title)}-会议纪要.md`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function renderTodoItems(items: string[], scope: string) {
    const visibleItems = items.filter((item) => item !== "暂无");
    if (!visibleItems.length) return <p>暂无</p>;
    const remainingItems = visibleItems.filter((item) => !todoInteractions[`${scope}:${item}`]?.removed);
    if (!remainingItems.length) return <p>待办已清空</p>;
    return (
      <ul className="todo-list">
        {remainingItems.map((item) => {
          const key = `${scope}:${item}`;
          const state = todoInteractions[key] || {};
          return (
            <li className={`todo-item ${state.done ? "done" : ""}`} key={key}>
              <button className="todo-check" type="button" aria-label={state.done ? "标记为未完成" : "标记为已完成"} onClick={() => updateTodoInteraction(key, { done: !state.done })}>
                {state.done && <Check size={13} />}
              </button>
              <span>{item}</span>
              <button className="icon-button subtle" type="button" aria-label="删除待办" onClick={() => updateTodoInteraction(key, { removed: true })}>
                <Trash2 size={15} />
              </button>
            </li>
          );
        })}
      </ul>
    );
  }

  function renderStructuredTodos(items: SummaryTodo[]) {
    if (!items.length) return <p className="summary-muted">暂无明确待办</p>;
    const visibleItems = items.filter((item) => !todoInteractions[`cards:${item.owner}:${item.task}:${item.due}`]?.removed);
    if (!visibleItems.length) return <p className="summary-muted">待办已清空</p>;
    return (
      <ul className="task-board">
        {visibleItems.map((item) => {
          const key = `cards:${item.owner}:${item.task}:${item.due}`;
          const state = todoInteractions[key] || {};
          return (
            <li className={`task-card ${state.done ? "done" : ""}`} key={key}>
              <button className="todo-check" type="button" aria-label={state.done ? "标记为未完成" : "标记为已完成"} onClick={() => updateTodoInteraction(key, { done: !state.done })}>
                {state.done && <Check size={13} />}
              </button>
              <div>
                <span>{item.task}</span>
                <small>{item.owner || "待补充"} · {item.due || "待补充"}</small>
              </div>
              <button className="icon-button subtle" type="button" aria-label="删除待办" onClick={() => updateTodoInteraction(key, { removed: true })}>
                <Trash2 size={15} />
              </button>
            </li>
          );
        })}
      </ul>
    );
  }

  function renderCardSummary() {
    const cards = displaySummaryCards;
    const todoCount = cards.todos.filter((item) => !todoInteractions[`cards:${item.owner}:${item.task}:${item.due}`]?.removed).length;
    return (
      <div className="cards-summary">
        <section className="overview-card">
          <div>
            <span>总览</span>
            <p>{cards.overview || "暂无一句话总览"}</p>
          </div>
          <div className="summary-stats" aria-label="纪要统计">
            <strong>结论 {cards.conclusions.length}</strong>
            <strong>待办 {todoCount}</strong>
            <strong>问题 {cards.risks.length}</strong>
          </div>
        </section>

        <div className="summary-main-grid">
          <section className="summary-block conclusion-block">
            <div className="block-head">
              <h3>结论</h3>
              <span>{cards.conclusions.length || "暂无"}</span>
            </div>
            {cards.conclusions.length ? (
              <ul className="decision-list">
                {cards.conclusions.map((item, index) => (
                  <li key={`conclusion-card-${index}`}>
                    <span>{item.status || "已确定"}</span>
                    <p>{item.text}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="summary-muted">暂无明确结论</p>
            )}
          </section>

          <section className="summary-block todo-block">
            <div className="block-head">
              <h3>待办</h3>
              <span>{todoCount || "暂无"}</span>
            </div>
            {renderStructuredTodos(cards.todos)}
          </section>
        </div>

        <section className="summary-block risk-block">
          <div className="block-head">
            <h3>问题</h3>
            <span>{cards.risks.length || "暂无"}</span>
          </div>
          {cards.risks.length ? (
            <div className="risk-list">
              {cards.risks.map((item, index) => (
                <article className={`risk-item level-${item.level || "中"}`} key={`risk-${index}`}>
                  <strong>{item.level || "中"}</strong>
                  <p>{item.text}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="summary-muted">暂未识别到明确风险或问题</p>
          )}
        </section>

        <section className="summary-block detail-block">
          <div className="block-head">
            <h3>详情</h3>
            <span>{cards.details.length || "暂无"}</span>
          </div>
          {cards.details.length ? (
            <div className="detail-groups">
              {cards.details.map((group, index) => (
                <details key={`detail-${index}`} open={index === 0}>
                  <summary>{group.title}</summary>
                  <ul>
                    {group.items.map((item, itemIndex) => <li key={`detail-${index}-${itemIndex}`}>{item}</li>)}
                  </ul>
                </details>
              ))}
            </div>
          ) : (
            <p className="summary-muted">暂无详细摘要</p>
          )}
        </section>
      </div>
    );
  }

  async function loadSettings() {
    const data = await api<AppSettings>("/settings");
    setSettingsDraft(data);
  }

  async function saveSettings() {
    await runBusy("settings", async () => {
      const saved = await api<AppSettings>("/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settingsDraft),
      });
      setSettingsDraft(saved);
      setSettingsMessage("设置已保存");
      window.setTimeout(() => setSettingsMessage(""), 2200);
    });
  }

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    loadMeetings().catch((err) => setError(String(err)));
    loadSettings().catch((err) => setError(String(err)));
    return () => {
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      monitorGainRef.current?.disconnect();
      audioContextRef.current?.close();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (selectedId) {
      loadDetail(selectedId).catch((err) => setError(String(err)));
    }
  }, [selectedId]);

  useEffect(() => {
    if (!detail) {
      setTodoInteractions({});
      return;
    }
    const raw = window.localStorage.getItem(`meeting-todos-${detail.meeting.id}`);
    if (!raw) {
      setTodoInteractions({});
      return;
    }
    try {
      setTodoInteractions(JSON.parse(raw) as Record<string, TodoInteraction>);
    } catch {
      setTodoInteractions({});
    }
  }, [detail?.meeting.id]);

  useEffect(() => {
    if (!detail) return;
    window.localStorage.setItem(`meeting-todos-${detail.meeting.id}`, JSON.stringify(todoInteractions));
  }, [detail?.meeting.id, todoInteractions]);

  useEffect(() => {
    if (!isRecording || !detail) return;
    const timer = window.setInterval(() => {
      generateStageSummaries(true).catch((err) => setError(err instanceof Error ? err.message : String(err)));
    }, 120000);
    return () => window.clearInterval(timer);
  }, [isRecording, detail?.meeting.id]);

  return (
    <main className={`app ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="nav-rail" aria-label="主导航">
        <nav className="rail-nav">
          <button className={`rail-button ${!settingsOpen ? "active" : ""}`} type="button" title="首页" aria-label="首页" onClick={() => { setSettingsOpen(false); loadMeetings().catch((err) => setError(String(err))); }}>
            <Home size={20} />
          </button>
          <button
            className="rail-button"
            type="button"
            title="上传音频"
            aria-label="上传音频"
            onClick={() => uploadFileInputRef.current?.click()}
            disabled={!detail || busy === "upload" || isRecording || isStartingRecording}
          >
            {busy === "upload" ? <Loader2 className="spin" size={20} /> : <FileAudio size={20} />}
          </button>
          <input
            ref={uploadFileInputRef}
            className="rail-file-input"
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.flac,.webm,.mp4"
            disabled={!detail || busy === "upload" || isRecording || isStartingRecording}
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.currentTarget.value = "";
              if (file) {
                setSettingsOpen(false);
                uploadAudio(file);
              }
            }}
          />
          {!isRecording ? (
            <button
              className="rail-record"
              type="button"
              title={isStartingRecording ? "启动中" : "开始会议"}
              aria-label={isStartingRecording ? "启动中" : "开始会议"}
              onClick={() => { setSettingsOpen(false); startNewMeetingRecording().catch((err) => setError(String(err))); }}
              disabled={Boolean(busy) || !microphoneReady || isStartingRecording}
            >
              {isStartingRecording ? <Loader2 className="spin" size={18} /> : <Mic size={18} />}
            </button>
          ) : (
            <button className="rail-record recording" type="button" title="停止录音" aria-label="停止录音" onClick={stopRecording}>
              <Square size={17} />
            </button>
          )}
          <div className="rail-spacer" />
          <button
            className={`rail-button ${settingsOpen ? "active" : ""}`}
            type="button"
            title="设置"
            aria-label="设置"
            onClick={() => setSettingsOpen(true)}
          >
            <Settings size={20} />
          </button>
        </nav>
      </aside>

      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="brand">
          <img className="brand-mascot" src="/su-xiaozhi.png" alt="苏小智" />
          {!sidebarCollapsed && (
            <div className="brand-copy">
              <strong>会议记录</strong>
              <span>苏小智 · 实时转写 / 智能纪要</span>
            </div>
          )}
          <button
            className="sidebar-toggle"
            type="button"
            title={sidebarCollapsed ? "展开会议列表" : "收起会议列表"}
            aria-label={sidebarCollapsed ? "展开会议列表" : "收起会议列表"}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
        </div>

        {!sidebarCollapsed && (
          <>
            <div className="create-box">
              <div className="search-row">
                <input
                  ref={meetingSearchInputRef}
                  value={meetingSearch}
                  onChange={(event) => setMeetingSearch(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") createMeeting();
                  }}
                  placeholder="搜索或输入会议名称"
                />
                <button type="button" title="搜索" aria-label="搜索会议" onClick={() => meetingSearchInputRef.current?.focus()}>
                  <Search size={17} />
                </button>
              </div>
              <button className="primary" onClick={createMeeting} disabled={busy === "create" || !meetingSearch.trim()}>
                {busy === "create" ? <Loader2 className="spin" size={17} /> : <Plus size={17} />}
                创建会议
              </button>
            </div>

            <div className="meeting-list">
              {filteredMeetings.map((meeting) => (
                <div
                  className={`meeting-item ${meeting.id === selectedId ? "active" : ""}`}
                  key={meeting.id}
                >
                  <button
                    className="meeting-select"
                    type="button"
                    disabled={isRecording || isStartingRecording}
                    onClick={() => {
                      setUploadText("");
                      setUploadSegments([]);
                      setStreamText("");
                      setStreamSegments([]);
                      setActiveView("transcript");
                      setSettingsOpen(false);
                      setSelectedId(meeting.id);
                    }}
                  >
                    <span>{meeting.title}</span>
                    <small>{meeting.status} / {formatTime(meeting.created_at)}</small>
                  </button>
                  <button
                    className="meeting-delete"
                    type="button"
                    title="删除会议"
                    aria-label={`删除会议 ${meeting.title}`}
                    disabled={Boolean(busy) || isRecording || isStartingRecording}
                    onClick={() => deleteMeeting(meeting).catch((err) => setError(String(err)))}
                  >
                    {busy === `delete-${meeting.id}` ? <Loader2 className="spin" size={16} /> : <Trash2 size={16} />}
                  </button>
                </div>
              ))}
              {meetings.length === 0 && <div className="empty-list">暂无会议</div>}
              {meetings.length > 0 && filteredMeetings.length === 0 && <div className="empty-list">未找到会议</div>}
            </div>
          </>
        )}
      </aside>

      <section className="workspace">
        {settingsOpen ? (
          <section className="settings-page" aria-label="设置">
            <header className="settings-page-head">
              <div>
                <h1>设置</h1>
                <p>配置转写、录音文件、智能纪要和实验能力。</p>
              </div>
              <div className="settings-actions top-actions">
                {settingsMessage && <span>{settingsMessage}</span>}
                <button type="button" onClick={() => loadSettings().catch((err) => setError(String(err)))} disabled={busy === "settings"}>重载</button>
                <button type="button" className="primary" onClick={saveSettings} disabled={busy === "settings"}>
                  {busy === "settings" ? <Loader2 className="spin" size={17} /> : null}
                  保存设置
                </button>
              </div>
            </header>

            <div className="settings-tabs settings-tabs-wide" role="tablist" aria-label="设置分类">
              <button type="button" className={settingsTab === "general" ? "active" : ""} onClick={() => setSettingsTab("general")}>
                <SlidersHorizontal size={16} />
                General
              </button>
              <button type="button" className={settingsTab === "recordings" ? "active" : ""} onClick={() => setSettingsTab("recordings")}>
                <Mic size={16} />
                Recordings
              </button>
              <button type="button" className={settingsTab === "transcript" ? "active" : ""} onClick={() => setSettingsTab("transcript")}>
                <Database size={16} />
                Transcription
              </button>
              <button type="button" className={settingsTab === "summary" ? "active" : ""} onClick={() => setSettingsTab("summary")}>
                <Sparkles size={16} />
                Summary
              </button>
              <button type="button" className={settingsTab === "beta" ? "active" : ""} onClick={() => setSettingsTab("beta")}>
                <FlaskConical size={16} />
                Beta
              </button>
            </div>

            <section className="settings-content">
              {settingsTab === "general" && (
                <div className="settings-section">
                  <div className="settings-section-head">
                    <h2>General</h2>
                    <p>应用级状态和数据保存位置。</p>
                  </div>
                  <div className="settings-grid two-col">
                    <div className="settings-info-card">
                      <strong>会议数据库</strong>
                      <span>backend/storage/db.json</span>
                      <p>保存会议列表、转写片段、智能摘要和当前设置。</p>
                    </div>
                    <div className="settings-info-card">
                      <strong>音频文件目录</strong>
                      <span>backend/storage/uploads</span>
                      <p>上传音频、浏览器录音和转 ASR 用的 wav 都在这里。</p>
                    </div>
                    <div className="settings-info-card">
                      <strong>API Base</strong>
                      <span>{API_BASE}</span>
                      <p>前端通过 Vite 代理访问后端。</p>
                    </div>
                    <div className="settings-info-card">
                      <strong>浏览器采样率</strong>
                      <span>{TARGET_SAMPLE_RATE / 1000} kHz</span>
                      <p>实时录音会下采样后发送给流式 ASR。</p>
                    </div>
                  </div>
                </div>
              )}

              {settingsTab === "recordings" && (
                <div className="settings-section">
                  <div className="settings-section-head">
                    <h2>Recordings</h2>
                    <p>当前网页版本的音频保存规则。</p>
                  </div>
                  <div className="settings-info-card full-width">
                    <div className="settings-card-title">
                      <HardDrive size={20} />
                      <strong>存储格式</strong>
                    </div>
                    <p>上传原文件会按原扩展名保存，例如 m4a、wav、webm。实时录音通常由浏览器保存为 webm，后端会额外转成 wav 给 ASR 使用。</p>
                    <p>当前代码没有生成 MP4 文件；如果你上传 mp4，后端会按上传原文件保留，但前端文件选择目前主要面向音频格式。</p>
                  </div>
                  <div className="settings-note">
                    <strong>文件名规则</strong>
                    <p>{"<meeting_id>-upload-<uuid>.<ext> / <meeting_id>-stream-<uuid>.webm / <meeting_id>-*-asr.wav"}</p>
                  </div>
                </div>
              )}

              {settingsTab === "transcript" && (
                <div className="settings-section">
                  <div className="settings-section-head">
                    <h2>Transcript Model</h2>
                    <p>截图里是桌面版的 Parakeet Lightning/Compact；当前网页后端实际使用 Qwen3-ASR。</p>
                  </div>
                  <div className="model-card selected">
                    <div>
                      <strong>Qwen3-ASR</strong>
                      <p>上传音频走 OpenAI-compatible transcription endpoint；实时录音走 streaming ASR 服务。</p>
                    </div>
                    <span>Active</span>
                  </div>
                  <div className="settings-form settings-form-grid">
                    <label>
                      <span>ASR 服务地址</span>
                      <input value={settingsDraft.asr_base_url} onChange={(event) => setSettingsDraft({ ...settingsDraft, asr_base_url: event.target.value })} placeholder="http://127.0.0.1:8005" />
                    </label>
                    <label>
                      <span>上传转写模型</span>
                      <input value={settingsDraft.asr_model} onChange={(event) => setSettingsDraft({ ...settingsDraft, asr_model: event.target.value })} />
                    </label>
                    <label>
                      <span>ASR API Key</span>
                      <input type="password" value={settingsDraft.asr_api_key} onChange={(event) => setSettingsDraft({ ...settingsDraft, asr_api_key: event.target.value })} />
                    </label>
                    <label>
                      <span>转写接口路径</span>
                      <input value={settingsDraft.asr_transcribe_path} onChange={(event) => setSettingsDraft({ ...settingsDraft, asr_transcribe_path: event.target.value })} />
                    </label>
                    <label>
                      <span>失败重试次数</span>
                      <input type="number" min="0" max="10" value={settingsDraft.asr_max_retries} onChange={(event) => setSettingsDraft({ ...settingsDraft, asr_max_retries: Number(event.target.value) })} />
                    </label>
                    <label>
                      <span>实时 ASR 地址</span>
                      <input value={settingsDraft.asr_stream_base_url} onChange={(event) => setSettingsDraft({ ...settingsDraft, asr_stream_base_url: event.target.value })} placeholder="http://127.0.0.1:8005" />
                    </label>
                  </div>
                </div>
              )}

              {settingsTab === "summary" && (
                <div className="settings-section">
                  <div className="settings-section-head">
                    <h2>Summary</h2>
                    <p>配置生成智能纪要使用的大模型服务。</p>
                  </div>
                  <div className="settings-form settings-form-grid">
                    <label>
                      <span>LLM 服务地址</span>
                      <input value={settingsDraft.llm_base_url} onChange={(event) => setSettingsDraft({ ...settingsDraft, llm_base_url: event.target.value })} placeholder="http://127.0.0.1:8012/v1" />
                    </label>
                    <label>
                      <span>纪要模型</span>
                      <input value={settingsDraft.llm_model} onChange={(event) => setSettingsDraft({ ...settingsDraft, llm_model: event.target.value })} />
                    </label>
                    <label>
                      <span>LLM API Key</span>
                      <input type="password" value={settingsDraft.llm_api_key} onChange={(event) => setSettingsDraft({ ...settingsDraft, llm_api_key: event.target.value })} />
                    </label>
                    <label>
                      <span>Temperature</span>
                      <input type="number" min="0" max="2" step="0.1" value={settingsDraft.llm_temperature} onChange={(event) => setSettingsDraft({ ...settingsDraft, llm_temperature: Number(event.target.value) })} />
                    </label>
                    <label>
                      <span>最大 Token</span>
                      <input type="number" min="128" max="16000" step="128" value={settingsDraft.llm_max_tokens} onChange={(event) => setSettingsDraft({ ...settingsDraft, llm_max_tokens: Number(event.target.value) })} />
                    </label>
                  </div>
                </div>
              )}

              {settingsTab === "beta" && (
                <div className="settings-section">
                  <div className="settings-section-head">
                    <h2>Beta</h2>
                    <p>后续可以接入的实验功能。</p>
                  </div>
                  <div className="settings-info-card full-width">
                    <strong>Forced Aligner 时间戳</strong>
                    <span>ASR_FORCED_ALIGNER</span>
                    <p>Qwen3-ASR 的流式模式不返回真实 timestamp；上传音频若配置 Qwen3-ForcedAligner，可获得更准确的词/片段时间戳。当前实时分段时间是后端按录音时长估算。</p>
                  </div>
                </div>
              )}
            </section>
          </section>
        ) : (
          <>
            <header className="topbar">
              <div className="meeting-title-line">
                <h1>{detail?.meeting.title || "创建会议后开始转写"}</h1>
                <p>{detail ? detail.meeting.status : "不会预置示例数据，页面只显示你的真实测试结果。"}</p>
              </div>
              <div className="mode-tabs" role="tablist" aria-label="工作区视图">
                <button
                  className={`mode-tab ${activeView === "transcript" ? "active" : ""}`}
                  type="button"
                  role="tab"
                  aria-selected={activeView === "transcript"}
                  onClick={() => setActiveView("transcript")}
                >
                  <Mic size={16} />
                  录音转写
                </button>
                <button
                  className={`mode-tab ${activeView === "summary" ? "active" : ""}`}
                  type="button"
                  role="tab"
                  aria-selected={activeView === "summary"}
                  onClick={() => setActiveView("summary")}
                >
                  <Sparkles size={16} />
                  智能纪要
                </button>
              </div>
            </header>

            {error && <div className="error">{error}</div>}

            <section className="work-grid">
              {activeView === "transcript" ? (
                <section className="panel focus-panel live-panel">
                  <div className="panel-head focus-head">
                    <div>
                      <h2>实时转写</h2>
                      <span>
                        {isRecording
                          ? "麦克风录音中，约 500ms 推送一次"
                          : isStartingRecording
                            ? "正在启动实时转写"
                            : microphoneReady
                            ? "点击开始后授权麦克风"
                            : "麦克风需要 HTTPS、localhost 或浏览器企业策略"}
                      </span>
                    </div>
                    <div className={`record-dot ${isRecording ? "on" : ""}`} />
                  </div>
                  {!microphoneReady && (
                    <div className="hint">浏览器默认不允许 HTTP 服务器 IP 页面调用麦克风。内网多人使用建议配置 HTTPS，或由 IT 下发浏览器企业策略。</div>
                  )}
                  <div className="action-row">
                    {!isRecording ? (
                      <button className="primary" onClick={() => startRecording()} disabled={!detail || Boolean(busy) || !microphoneReady || isStartingRecording}>
                        {isStartingRecording ? <Loader2 className="spin" size={17} /> : <Mic size={17} />}
                        {isStartingRecording ? "启动中" : "开始会议"}
                      </button>
                    ) : (
                      <button className="danger" onClick={stopRecording}>
                        <Square size={16} />
                        停止
                      </button>
                    )}
                  </div>
                  <div className="transcript-feed" aria-label="转写记录">
                    {transcriptRows.map((segment, index) => (
                      <article className="transcript-row" key={segment.id || `${segment.start}-${index}`}>
                        <time>{formatSegmentTime(segment.start)}</time>
                        <div>
                          <strong>发言人</strong>
                          <p>{segment.text}</p>
                        </div>
                      </article>
                    ))}
                    {transcriptRows.length === 0 && (
                      <div className="empty-transcript">现在就可以开始录音了</div>
                    )}
                  </div>
                </section>
              ) : (
                <section className="panel focus-panel summary-panel">
                  <div className="panel-head focus-head">
                    <div>
                      <h2>智能纪要</h2>
                      <span>阶段摘要、卡片纪要和正式文档</span>
                    </div>
                    <div className="summary-actions">
                      {summaryTab === "timeline" && (
                        <button className="primary" onClick={() => generateStageSummaries()} disabled={!detail || busy === "stage-summary" || !activeText.trim()}>
                          {busy === "stage-summary" ? <Loader2 className="spin" size={17} /> : <Clock3 size={17} />}
                          生成阶段摘要
                        </button>
                      )}
                      {summaryTab === "cards" && (
                        <button className="primary" onClick={generateSummary} disabled={!detail || busy === "summary" || isRecording || isStartingRecording || !activeText.trim()}>
                          {busy === "summary" ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
                          生成卡片纪要
                        </button>
                      )}
                      {summaryTab === "document" && (
                        <>
                          <button className="primary" onClick={generateMinutesDocument} disabled={!detail || busy === "minutes-document" || isRecording || isStartingRecording || !activeText.trim()}>
                            {busy === "minutes-document" ? <Loader2 className="spin" size={17} /> : <FileText size={17} />}
                            生成正式文档
                          </button>
                          <button onClick={copyMinutesDocument} disabled={!minutesDocument.trim()}>
                            <Clipboard size={16} />
                            复制
                          </button>
                          <button onClick={downloadMinutesDocument} disabled={!minutesDocument.trim()}>
                            <Download size={16} />
                            下载
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="summary-tabs" role="tablist" aria-label="纪要类型">
                    <button className={summaryTab === "timeline" ? "active" : ""} type="button" role="tab" aria-selected={summaryTab === "timeline"} onClick={() => setSummaryTab("timeline")}>
                      <Clock3 size={15} />
                      阶段摘要
                    </button>
                    <button className={summaryTab === "cards" ? "active" : ""} type="button" role="tab" aria-selected={summaryTab === "cards"} onClick={() => setSummaryTab("cards")}>
                      <Sparkles size={15} />
                      卡片纪要
                    </button>
                    <button className={summaryTab === "document" ? "active" : ""} type="button" role="tab" aria-selected={summaryTab === "document"} onClick={() => setSummaryTab("document")}>
                      <FileText size={15} />
                      正式文档
                    </button>
                  </div>
                  <div className="summary-content" aria-label="智能纪要内容">
                    {summaryTab === "timeline" && (
                      stageSummaries.length ? (
                        <div className="timeline-list">
                          {stageSummaries.map((item) => (
                            <article className="timeline-card" key={item.id}>
                              <div className="timeline-time">{formatTimeRange(item.start, item.end)}</div>
                              <div className="timeline-body">
                                <h3>{item.title}</h3>
                                <div className="timeline-columns">
                                  <section>
                                    <strong>摘要</strong>
                                    <ul>
                                      {item.summary.map((line, index) => <li key={`summary-${item.id}-${index}`}>{line}</li>)}
                                    </ul>
                                  </section>
                                  <section>
                                    <strong>结论</strong>
                                    <ul>
                                      {item.conclusions.map((line, index) => <li key={`conclusion-${item.id}-${index}`}>{line}</li>)}
                                    </ul>
                                  </section>
                                  <section>
                                    <strong>待办</strong>
                                    {renderTodoItems(item.todos, `stage:${item.id}`)}
                                  </section>
                                </div>
                              </div>
                            </article>
                          ))}
                        </div>
                      ) : (
                        <div className="summary-empty">每两分钟会沉淀一张阶段摘要卡片，也可以手动生成。</div>
                      )
                    )}
                    {summaryTab === "cards" && (
                      hasSummaryContent ? (
                        renderCardSummary()
                      ) : (
                        <div className="summary-empty">生成后会在这里按模块展示纪要。</div>
                      )
                    )}
                    {summaryTab === "document" && (
                      minutesDocument.trim() ? (
                        <article className="document-preview">
                          {minutesDocument.split("\n").map((line, index) => (
                            <p className={line.startsWith("#") ? "document-heading" : ""} key={`document-line-${index}`}>
                              {cleanSummaryLine(line) || " "}
                            </p>
                          ))}
                        </article>
                      ) : (
                        <div className="summary-empty">生成后会在这里展示可交付的正式会议纪要。</div>
                      )
                    )}
                  </div>
                </section>
              )}
            </section>
          </>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
