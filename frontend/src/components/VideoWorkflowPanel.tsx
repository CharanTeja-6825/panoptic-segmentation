import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getJobStatus,
  getVideoAnalysis,
  processVideo,
  sendVideoChatMessage,
  uploadVideo,
} from "../services/api";

type JobState = "idle" | "uploaded" | "processing" | "done" | "failed";

interface VideoChatItem {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const STEP_LABELS: Record<JobState, string> = {
  idle: "Upload a video",
  uploaded: "Uploaded — starting…",
  processing: "Processing…",
  done: "Complete",
  failed: "Failed",
};

export default function VideoWorkflowPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState("");
  const [jobState, setJobState] = useState<JobState>("idle");
  const [progress, setProgress] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [chatItems, setChatItems] = useState<VideoChatItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const progressPercent = useMemo(() => {
    if (!totalFrames) return 0;
    return Math.min(100, Math.round((progress / totalFrames) * 100));
  }, [progress, totalFrames]);

  useEffect(() => {
    if (!jobId || jobState !== "processing") {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const status = await getJobStatus(jobId);
        setProgress(Number(status.progress ?? 0));
        setTotalFrames(Number(status.total_frames ?? 0));
        setFps(Number(status.fps ?? 0));

        if (status.status === "done") {
          setJobState("done");
          window.clearInterval(timer);
          const artifact = await getVideoAnalysis(jobId);
          setAnalysis(artifact.analysis);
        }

        if (status.status === "failed") {
          setJobState("failed");
          setError(String(status.error ?? "Processing failed"));
          window.clearInterval(timer);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unable to poll job status");
        setJobState("failed");
        window.clearInterval(timer);
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [jobId, jobState]);

  const startWorkflow = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setAnalysis(null);
    setChatItems([]);
    setProgress(0);
    setTotalFrames(0);
    setFps(0);
    try {
      const upload = await uploadVideo(file);
      setJobId(upload.job_id);
      setJobState("uploaded");
      await processVideo(upload.job_id);
      setJobState("processing");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Video upload/process failed");
      setJobState("failed");
    } finally {
      setBusy(false);
    }
  };

  const sendQuestion = async () => {
    const text = chatInput.trim();
    if (!text || !jobId || jobState !== "done" || chatBusy) return;
    const userMsg: VideoChatItem = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
    };
    setChatItems((items) => [...items, userMsg]);
    setChatInput("");
    setChatBusy(true);
    try {
      const reply = await sendVideoChatMessage(jobId, {
        message: text,
        max_keyframes: 3,
      });
      setChatItems((items) => [
        ...items,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: reply.reply,
        },
      ]);
    } catch (e) {
      setChatItems((items) => [
        ...items,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: `Error: ${e instanceof Error ? e.message : "Video Q&A failed"}`,
        },
      ]);
    } finally {
      setChatBusy(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  const analysisSummary = (analysis?.summary as Record<string, unknown>) ?? null;
  const keyframeCount = Number(analysisSummary?.keyframes_count ?? 0);
  const eventCount = Number(analysisSummary?.events_count ?? 0);

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Processed Video Studio</h2>
          <p className="mt-0.5 text-xs" style={{ color: "var(--color-text-muted)" }}>
            Full-video segmentation, artifacts &amp; grounded Q&amp;A
          </p>
        </div>
        <span className={`status-pill ${jobState === "done" ? "status-pill-success" : jobState === "failed" ? "status-pill-danger" : "status-pill-muted"}`}>
          {STEP_LABELS[jobState]}
        </span>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {/* Upload zone */}
        <div
          className="relative mb-4 rounded-xl p-6 text-center transition-all"
          style={{
            background: isDragging ? "var(--color-accent-glow)" : "var(--color-bg-secondary)",
            border: isDragging ? "2px dashed var(--color-accent)" : "2px dashed var(--color-border)",
            cursor: "pointer",
          }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="hidden"
          />
          <svg className="mx-auto mb-3 h-8 w-8" style={{ color: "var(--color-text-muted)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
            {file ? file.name : "Drop a video or click to upload"}
          </p>
          <p className="mt-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
            MP4, AVI, MOV, MKV · Max 500 MB
          </p>
        </div>

        {/* Actions */}
        <div className="mb-4 flex gap-3">
          <button
            onClick={startWorkflow}
            disabled={!file || busy}
            className="btn btn-primary flex-1"
          >
            {busy ? (
              <>
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Submitting…
              </>
            ) : (
              <>
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                </svg>
                Upload &amp; Process
              </>
            )}
          </button>
          <a
            href={jobId && jobState === "done" ? `/api/download/${jobId}` : "#"}
            className={`btn flex-1 ${jobId && jobState === "done" ? "btn-primary" : "btn-ghost pointer-events-none opacity-50"}`}
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            Download Output
          </a>
        </div>

        {/* Progress */}
        {(jobState === "processing" || jobState === "done") && (
          <div className="mb-4">
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
              <span>{progress} / {totalFrames || "?"} frames</span>
              <span>FPS: {fps.toFixed(1)}</span>
              <span>Events: {eventCount}</span>
              <span>Keyframes: {keyframeCount}</span>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 rounded-lg px-3 py-2 text-sm" style={{ background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.2)", color: "#fca5a5" }}>
            {error}
          </div>
        )}

        {/* Q&A Section */}
        <div className="rounded-xl p-4" style={{ background: "var(--color-bg-secondary)", border: "1px solid var(--color-border)" }}>
          <p className="mb-3 text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
            Ask about processed video
          </p>
          <div className="mb-3 max-h-56 space-y-2 overflow-y-auto rounded-lg p-2" style={{ background: "var(--color-bg-primary)", border: "1px solid var(--color-border)" }}>
            {chatItems.length === 0 ? (
              <p className="py-3 text-center text-xs" style={{ color: "var(--color-text-muted)" }}>
                Process a video, then ask questions about activity, counts, or timelines.
              </p>
            ) : (
              chatItems.map((item) => (
                <div
                  key={item.id}
                  className={item.role === "user" ? "chat-bubble chat-bubble-user ml-auto" : "chat-bubble chat-bubble-assistant"}
                  style={{ maxWidth: "90%" }}
                >
                  {item.content}
                </div>
              ))
            )}
          </div>
          <div className="flex gap-2">
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void sendQuestion();
                }
              }}
              placeholder="Ask anything about this processed video…"
              disabled={jobState !== "done" || chatBusy}
              className="input-field h-10 flex-1"
            />
            <button
              onClick={() => void sendQuestion()}
              disabled={jobState !== "done" || chatBusy || !chatInput.trim()}
              className="btn btn-primary h-10"
            >
              {chatBusy ? "Asking…" : "Ask"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
