import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { FileAudio, Loader2, Mic, Plus, RefreshCw, Sparkles, Square, Upload } from "lucide-react";
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
};

type TranscriptionResult = {
  text: string;
  segments: Segment[];
};

type SummaryResult = {
  summary: string;
};

type StreamStartResult = {
  session_id: string;
};

const API_BASE = "/api";
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

function transcriptText(segments: Segment[]) {
  return segments.map((item) => item.text).filter(Boolean).join("\n");
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
  const [title, setTitle] = useState("");
  const [uploadText, setUploadText] = useState("");
  const [streamText, setStreamText] = useState("");
  const [summaryText, setSummaryText] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isStartingRecording, setIsStartingRecording] = useState(false);
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

  const currentTranscript = useMemo(() => transcriptText(detail?.transcript || []), [detail]);
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
    const nextId = nextSelectedId || selectedId || data[0]?.id || "";
    setSelectedId(nextId);
    if (nextId) {
      await loadDetail(nextId);
    } else {
      setDetail(null);
      setUploadText("");
      setStreamText("");
      setSummaryText("");
    }
  }

  async function loadDetail(id = selectedId) {
    if (!id) return;
    const nextDetail = await api<MeetingDetail>(`/meetings/${id}`);
    setDetail(nextDetail);
    setSummaryText(nextDetail.summary || "");
  }

  async function createMeeting() {
    if (!title.trim()) return;
    await runBusy("create", async () => {
      const meeting = await api<Meeting>("/meetings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim() }),
      });
      setTitle("");
      setUploadText("");
      setStreamText("");
      setSummaryText("");
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
      setStreamText("");
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

  async function startRecording() {
    if (!detail || isRecording || isStartingRecording) return;
    setError("");
    if (!microphoneReady) {
      setError("当前访问地址不能使用麦克风。请使用 HTTPS、localhost，或让公司 IT 通过浏览器策略把当前 HTTP 内网地址加入安全源。");
      return;
    }
    setUploadText("");
    setStreamText("");
    bufferedSamplesRef.current = [];
    pendingChunksRef.current = [];
    isPushingRef.current = false;
    setIsStartingRecording(true);
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const started = await api<StreamStartResult>(`/meetings/${detail.meeting.id}/stream/start`, { method: "POST" });
      streamMeetingIdRef.current = detail.meeting.id;
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

  async function generateSummary() {
    if (!detail) return;
    await runBusy("summary", async () => {
      const result = await api<SummaryResult>(`/meetings/${detail.meeting.id}/summary`, { method: "POST" });
      setSummaryText(result.summary);
      await loadDetail(detail.meeting.id);
    });
  }

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    loadMeetings().catch((err) => setError(String(err)));
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

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <Mic size={26} />
          <div>
            <strong>AI 会议转写</strong>
            <span>实时转写 / 音频上传 / AI 纪要</span>
          </div>
        </div>

        <div className="create-box">
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") createMeeting();
            }}
            placeholder="输入会议名称"
          />
          <button className="primary" onClick={createMeeting} disabled={busy === "create" || !title.trim()}>
            {busy === "create" ? <Loader2 className="spin" size={17} /> : <Plus size={17} />}
            创建会议
          </button>
        </div>

        <div className="meeting-list">
          {meetings.map((meeting) => (
            <button
              className={`meeting-item ${meeting.id === selectedId ? "active" : ""}`}
              key={meeting.id}
              disabled={isRecording || isStartingRecording}
              onClick={() => {
                setUploadText("");
                setStreamText("");
                setSelectedId(meeting.id);
              }}
            >
              <span>{meeting.title}</span>
              <small>{meeting.status} / {formatTime(meeting.created_at)}</small>
            </button>
          ))}
          {meetings.length === 0 && <div className="empty-list">暂无会议</div>}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>{detail?.meeting.title || "创建会议后开始转写"}</h1>
            <p>{detail ? detail.meeting.status : "不会预置示例数据，页面只显示你的真实测试结果。"}</p>
          </div>
          <button onClick={() => loadMeetings()} disabled={Boolean(busy) || isRecording || isStartingRecording}>
            <RefreshCw size={17} />
            刷新
          </button>
        </header>

        {error && <div className="error">{error}</div>}

        <section className="work-grid">
          <section className="panel live-panel">
            <div className="panel-head">
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
                <button className="primary" onClick={startRecording} disabled={!detail || Boolean(busy) || !microphoneReady || isStartingRecording}>
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
            <textarea className="output live-output" value={streamText} readOnly placeholder="实时转写文字会直接输出到这里。" />
          </section>

          <section className="panel upload-panel">
            <div className="panel-head">
              <div>
                <h2>上传音频</h2>
                <span>选择音频文件后立即转写</span>
              </div>
              <FileAudio size={22} />
            </div>
            <label className={`upload-zone ${busy === "upload" ? "busy" : ""}`}>
              {busy === "upload" ? <Loader2 className="spin" size={22} /> : <Upload size={22} />}
              <span>{busy === "upload" ? "正在转写音频" : "选择音频文件"}</span>
              <input
                type="file"
                accept="audio/*,.mp3,.wav,.m4a,.flac,.webm"
                disabled={!detail || busy === "upload" || isRecording || isStartingRecording}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  event.currentTarget.value = "";
                  if (file) uploadAudio(file);
                }}
              />
            </label>
            <textarea className="output" value={uploadText} readOnly placeholder="上传音频的转写文字会输出到这里。" />
          </section>

          <section className="panel transcript-panel">
            <div className="panel-head">
              <div>
                <h2>当前会议文字</h2>
                <span>{detail?.transcript.length || 0} 段</span>
              </div>
            </div>
            <textarea className="output transcript-output" value={activeText} readOnly placeholder="当前会议暂无转写结果。" />
          </section>

          <section className="panel summary-panel">
            <div className="panel-head">
              <div>
                <h2>AI 纪要总结</h2>
                <span>调用本地 vLLM 生成摘要、结论和待跟进事项</span>
              </div>
              <button className="primary" onClick={generateSummary} disabled={!detail || busy === "summary" || isRecording || isStartingRecording || !activeText.trim()}>
                {busy === "summary" ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
                生成纪要
              </button>
            </div>
            <textarea className="output summary-output" value={summaryText} readOnly placeholder="生成后的 AI 纪要会输出到这里。" />
          </section>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
