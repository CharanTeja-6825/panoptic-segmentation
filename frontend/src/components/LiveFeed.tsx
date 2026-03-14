import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store/useStore";
import { getCameraStreamUrl } from "../services/api";
import { eventSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";
import { normalizeSceneObject, summarizeObjectCounts } from "../lib/mappers";

export default function LiveFeed() {
  const cameraActive = useStore((s) => s.cameraActive);
  const sceneObjects = useStore((s) => s.sceneObjects);
  const setSceneObjects = useStore((s) => s.setSceneObjects);
  const setSystemHealth = useStore((s) => s.setSystemHealth);
  const [streamError, setStreamError] = useState(false);

  useWebSocket(eventSocket, "message", (data) => {
    const msg = data as Record<string, unknown>;
    if (msg.type === "scene_update" && Array.isArray(msg.objects)) {
      const normalized = (msg.objects as Array<Record<string, unknown>>).map(
        normalizeSceneObject
      );
      setSceneObjects(normalized);
    }
  });

  useWebSocket(eventSocket, "connection", (data) => {
    const status = (data as { status?: string }).status;
    setSystemHealth({ websocket: status === "connected" });
  });

  useEffect(() => {
    eventSocket.connect();
  }, []);

  useEffect(() => {
    if (cameraActive) {
      setStreamError(false);
    }
  }, [cameraActive]);

  const streamUrl = useMemo(
    () => `${getCameraStreamUrl()}?t=${Date.now()}`,
    [cameraActive]
  );
  const objectCountMap = summarizeObjectCounts(sceneObjects);

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Live Scene Stream</h2>
          <p className="mt-0.5 text-xs text-slate-400">
            Segmentation, tracking IDs, and real-time object updates
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="status-pill status-pill-muted">
            {sceneObjects.length} Objects
          </span>
          {cameraActive ? (
            <span className="status-pill status-pill-success">
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-300" />
              Live
            </span>
          ) : (
            <span className="status-pill status-pill-muted">Idle</span>
          )}
        </div>
      </header>

      <div className="relative min-h-0 flex-1 bg-slate-950">
        {cameraActive && !streamError ? (
          <img
            src={streamUrl}
            alt="Live camera feed"
            className="h-full w-full object-contain"
            onError={() => setStreamError(true)}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <div className="rounded-full border border-slate-700 bg-slate-900 p-4">
              <svg
                className="h-8 w-8 text-slate-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
                />
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-300">
              {streamError ? "Stream connection lost" : "Camera feed inactive"}
            </p>
            <p className="text-xs text-slate-500">
              Start the camera in the control panel to begin live scene analysis.
            </p>
          </div>
        )}

        {cameraActive && Object.keys(objectCountMap).length > 0 && (
          <div className="absolute bottom-3 left-3 right-3 rounded-xl border border-slate-700/70 bg-slate-900/80 p-3 backdrop-blur-sm">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Current detections
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(objectCountMap)
                .sort(([, a], [, b]) => b - a)
                .map(([label, count]) => (
                  <span
                    key={label}
                    className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
                  >
                    <span className="font-semibold text-indigo-300">{count}</span>{" "}
                    {label}
                  </span>
                ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
