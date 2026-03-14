import { useEffect, useMemo, useState } from "react";
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

  const analysisSummary = (analysis?.summary as Record<string, unknown>) ?? null;
  const keyframeCount = Number(analysisSummary?.keyframes_count ?? 0);
  const eventCount = Number(analysisSummary?.events_count ?? 0);

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Processed Video Studio</h2>
          <p className="mt-0.5 text-xs text-slate-400">
            Full-video segmentation, artifact generation, and grounded Q&A
          </p>
        </div>
        <span className="status-pill status-pill-muted">{jobState}</span>
      </header>

      <div className="grid gap-3 p-4 md:grid-cols-3">
        <input
          type="file"
          accept=".mp4,.avi,.mov,.mkv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
        />
        <button
          onClick={startWorkflow}
          disabled={!file || busy}
          className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Submitting..." : "Upload & Process"}
        </button>
        <a
          href={jobId && jobState === "done" ? `/api/download/${jobId}` : "#"}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${
            jobId && jobState === "done"
              ? "bg-emerald-600 text-white hover:bg-emerald-500"
              : "pointer-events-none bg-slate-700 text-slate-400"
          }`}
        >
          Download Output
        </a>
      </div>

      <div className="space-y-2 px-4 pb-3">
        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full bg-indigo-500 transition-all"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <span>Progress: {progress} / {totalFrames || "?"} frames</span>
          <span>FPS: {fps.toFixed(1)}</span>
          <span>Events: {eventCount}</span>
          <span>Keyframes: {keyframeCount}</span>
        </div>
        {error ? <p className="text-xs text-rose-400">{error}</p> : null}
      </div>

      <div className="border-t border-slate-700/60 px-4 py-3">
        <p className="mb-2 text-xs uppercase tracking-wider text-slate-400">
          Ask about processed video
        </p>
        <div className="mb-2 max-h-48 space-y-2 overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 p-2">
          {chatItems.length === 0 ? (
            <p className="text-xs text-slate-500">
              Process a video, then ask questions about activity, counts, or timelines.
            </p>
          ) : (
            chatItems.map((item) => (
              <div
                key={item.id}
                className={`rounded-lg px-3 py-2 text-sm ${
                  item.role === "user"
                    ? "bg-indigo-600 text-white"
                    : "border border-slate-700 bg-slate-800 text-slate-100"
                }`}
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
            placeholder="Ask anything about this processed video..."
            disabled={jobState !== "done" || chatBusy}
            className="h-10 flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 text-sm text-slate-100"
          />
          <button
            onClick={() => void sendQuestion()}
            disabled={jobState !== "done" || chatBusy || !chatInput.trim()}
            className="h-10 rounded-lg bg-indigo-600 px-3 text-sm text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {chatBusy ? "Asking..." : "Ask"}
          </button>
        </div>
      </div>
    </section>
  );
}
