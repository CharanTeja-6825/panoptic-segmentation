import { useState } from "react";
import { useStore } from "../store/useStore";
import { startCamera, stopCamera } from "../services/api";

export default function ControlPanel() {
  const cameraActive = useStore((s) => s.cameraActive);
  const setCameraActive = useStore((s) => s.setCameraActive);
  const setSystemHealth = useStore((s) => s.setSystemHealth);
  const detectionSensitivity = useStore((s) => s.detectionSensitivity);
  const setDetectionSensitivity = useStore((s) => s.setDetectionSensitivity);
  const recording = useStore((s) => s.recording);
  const setRecording = useStore((s) => s.setRecording);
  const addSceneEvent = useStore((s) => s.addSceneEvent);
  const [busy, setBusy] = useState(false);

  const pushSystemEvent = (message: string, severity: "low" | "medium" | "high") => {
    addSceneEvent({
      id: `ui-${Date.now()}`,
      type: "system",
      message,
      timestamp: new Date().toISOString(),
      severity,
    });
  };

  const handleCameraToggle = async () => {
    setBusy(true);
    try {
      if (cameraActive) {
        await stopCamera();
        setCameraActive(false);
        setSystemHealth({ camera: false });
        pushSystemEvent("Camera stream stopped.", "low");
      } else {
        await startCamera();
        setCameraActive(true);
        setSystemHealth({ camera: true });
        pushSystemEvent("Camera stream started.", "low");
      }
    } catch {
      pushSystemEvent("Unable to change camera state. Check backend status.", "high");
    } finally {
      setBusy(false);
    }
  };

  const handleRecordToggle = () => {
    const next = !recording;
    setRecording(next);
    pushSystemEvent(next ? "Recording mode enabled." : "Recording mode disabled.", "low");
  };

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Controls</h2>
          <p className="mt-0.5 text-xs" style={{ color: "var(--color-text-muted)" }}>
            Camera, recording &amp; detection settings
          </p>
        </div>
      </header>

      <div className="grid gap-4 p-4 md:grid-cols-3">
        {/* Camera */}
        <div className="metric-card">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg" style={{ background: "var(--color-accent-glow)" }}>
              <svg className="h-4 w-4" style={{ color: "var(--color-accent-hover)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <span className="text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
              Camera
            </span>
          </div>
          <button
            onClick={handleCameraToggle}
            disabled={busy}
            className={cameraActive ? "btn btn-danger w-full" : "btn btn-primary w-full"}
          >
            {busy ? "Updating…" : cameraActive ? "Stop Camera" : "Start Camera"}
          </button>
        </div>

        {/* Recording */}
        <div className="metric-card">
          <div className="mb-3 flex items-center gap-2">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-lg"
              style={{ background: recording ? "rgba(248,113,113,0.15)" : "var(--color-accent-glow)" }}
            >
              <svg className="h-4 w-4" style={{ color: recording ? "var(--color-danger)" : "var(--color-accent-hover)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
            </div>
            <span className="text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
              Recording
            </span>
          </div>
          <button
            onClick={handleRecordToggle}
            className={recording ? "btn btn-danger w-full" : "btn btn-ghost w-full"}
          >
            {recording ? "Stop Recording" : "Start Recording"}
          </button>
        </div>

        {/* Sensitivity */}
        <div className="metric-card">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg" style={{ background: "var(--color-accent-glow)" }}>
              <svg className="h-4 w-4" style={{ color: "var(--color-accent-hover)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
              </svg>
            </div>
            <span className="text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
              Sensitivity
            </span>
          </div>
          <div className="mb-3 flex items-center justify-between text-sm">
            <span style={{ color: "var(--color-text-secondary)" }}>Threshold</span>
            <span className="font-bold text-lg" style={{ color: "var(--color-accent-hover)" }}>
              {detectionSensitivity}%
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={detectionSensitivity}
            onChange={(e) => setDetectionSensitivity(Number(e.target.value))}
            className="range-slider"
          />
        </div>
      </div>
    </section>
  );
}
