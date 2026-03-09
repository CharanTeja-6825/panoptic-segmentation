import { useEffect, useState } from "react";
import { useStore } from "../store/useStore";
import { getCameraStreamUrl } from "../services/api";
import { eventSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";

export default function LiveFeed() {
  const cameraActive = useStore((s) => s.cameraActive);
  const sceneObjects = useStore((s) => s.sceneObjects);
  const setSceneObjects = useStore((s) => s.setSceneObjects);
  const setSystemHealth = useStore((s) => s.setSystemHealth);
  const [streamError, setStreamError] = useState(false);

  useWebSocket(eventSocket, "message", (data) => {
    const msg = data as Record<string, unknown>;
    if (msg.type === "scene_update" && Array.isArray(msg.objects)) {
      setSceneObjects(
        (msg.objects as Array<Record<string, unknown>>).map((o) => ({
          id: String(o.id ?? Math.random()),
          label: String(o.label ?? "unknown"),
          confidence: Number(o.confidence ?? 0),
          bbox: (o.bbox as [number, number, number, number]) ?? [0, 0, 0, 0],
        }))
      );
    }
    if (msg.type === "connection") {
      setSystemHealth({ websocket: msg.status === "connected" });
    }
  });

  useEffect(() => {
    eventSocket.connect();
  }, []);

  const objectCountMap = sceneObjects.reduce<Record<string, number>>(
    (acc, obj) => {
      acc[obj.label] = (acc[obj.label] ?? 0) + 1;
      return acc;
    },
    {}
  );

  return (
    <div className="group relative flex flex-col overflow-hidden rounded-xl border border-slate-700/50 bg-slate-800 shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/50 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <svg
            className="h-4 w-4 text-indigo-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <h2 className="text-sm font-semibold text-slate-200">Live Feed</h2>
        </div>
        <div className="flex items-center gap-2">
          {cameraActive && (
            <span className="flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-400">
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-400" />
              LIVE
            </span>
          )}
          <span className="rounded-md bg-slate-700/50 px-2 py-0.5 text-xs text-slate-400">
            {sceneObjects.length} objects
          </span>
        </div>
      </div>

      {/* Video Area */}
      <div className="relative flex-1 bg-slate-900">
        {cameraActive && !streamError ? (
          <img
            src={getCameraStreamUrl()}
            alt="Live camera feed"
            className="h-full w-full object-contain"
            onError={() => setStreamError(true)}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 p-8">
            <div className="rounded-full bg-slate-800 p-4">
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
            <p className="text-sm text-slate-500">
              {streamError
                ? "Stream connection lost"
                : "Camera feed inactive"}
            </p>
            <p className="text-xs text-slate-600">
              Start the camera from controls below
            </p>
          </div>
        )}

        {/* Bounding Box Overlays */}
        {cameraActive &&
          sceneObjects.map((obj) => (
            <div
              key={obj.id}
              className="pointer-events-none absolute border-2 border-indigo-400/70"
              style={{
                left: `${obj.bbox[0]}%`,
                top: `${obj.bbox[1]}%`,
                width: `${obj.bbox[2] - obj.bbox[0]}%`,
                height: `${obj.bbox[3] - obj.bbox[1]}%`,
              }}
            >
              <span className="absolute -top-5 left-0 rounded bg-indigo-500/90 px-1.5 py-0.5 text-[10px] font-medium text-white">
                {obj.label} {(obj.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}

        {/* Object Count Overlay */}
        {cameraActive && Object.keys(objectCountMap).length > 0 && (
          <div className="absolute bottom-3 left-3 rounded-lg bg-black/60 px-3 py-2 backdrop-blur-sm">
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {Object.entries(objectCountMap).map(([label, count]) => (
                <span key={label} className="text-xs text-slate-200">
                  <span className="font-semibold text-indigo-300">{count}</span>{" "}
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
