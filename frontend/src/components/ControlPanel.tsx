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
          <p className="mt-0.5 text-xs text-slate-400">
            Manage camera, recording intent, and detection sensitivity
          </p>
        </div>
      </header>

      <div className="grid gap-3 p-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-700 bg-slate-900 p-3">
          <p className="mb-2 text-xs uppercase tracking-wider text-slate-400">Camera</p>
          <button
            onClick={handleCameraToggle}
            disabled={busy}
            className={`w-full rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
              cameraActive
                ? "bg-rose-600 text-white hover:bg-rose-500"
                : "bg-emerald-600 text-white hover:bg-emerald-500"
            } disabled:cursor-not-allowed disabled:opacity-60`}
          >
            {busy ? "Updating..." : cameraActive ? "Stop Camera" : "Start Camera"}
          </button>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-900 p-3">
          <p className="mb-2 text-xs uppercase tracking-wider text-slate-400">
            Recording
          </p>
          <button
            onClick={handleRecordToggle}
            className={`w-full rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
              recording
                ? "bg-rose-600 text-white hover:bg-rose-500"
                : "bg-slate-700 text-slate-200 hover:bg-slate-600"
            }`}
          >
            {recording ? "Stop Recording" : "Start Recording"}
          </button>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-900 p-3">
          <p className="mb-2 text-xs uppercase tracking-wider text-slate-400">
            Detection sensitivity
          </p>
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-slate-300">Threshold</span>
            <span className="font-semibold text-indigo-300">
              {detectionSensitivity}%
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={detectionSensitivity}
            onChange={(e) => setDetectionSensitivity(Number(e.target.value))}
            className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-indigo-500"
          />
        </div>
      </div>
    </section>
  );
}
