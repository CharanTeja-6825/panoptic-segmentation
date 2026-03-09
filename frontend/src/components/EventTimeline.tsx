import { useEffect, useState } from "react";
import { useStore, SceneEvent } from "../store/useStore";
import { getSceneEvents } from "../services/api";
import { eventSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";

type FilterType = "all" | SceneEvent["type"];

const EVENT_ICONS: Record<string, string> = {
  detection: "🔍",
  alert: "🚨",
  info: "ℹ️",
  warning: "⚠️",
  system: "⚙️",
};

const EVENT_COLORS: Record<string, string> = {
  detection: "border-indigo-500/30 bg-indigo-500/5",
  alert: "border-rose-500/30 bg-rose-500/5",
  info: "border-sky-500/30 bg-sky-500/5",
  warning: "border-amber-500/30 bg-amber-500/5",
  system: "border-slate-500/30 bg-slate-500/5",
};

const SEVERITY_DOT: Record<string, string> = {
  low: "bg-emerald-400",
  medium: "bg-amber-400",
  high: "bg-rose-400",
};

export default function EventTimeline() {
  const sceneEvents = useStore((s) => s.sceneEvents);
  const addSceneEvent = useStore((s) => s.addSceneEvent);
  const setSceneEvents = useStore((s) => s.setSceneEvents);
  const [filter, setFilter] = useState<FilterType>("all");

  useWebSocket(eventSocket, "message", (data) => {
    const msg = data as Record<string, unknown>;
    if (msg.type === "event" && msg.event) {
      const evt = msg.event as Record<string, unknown>;
      addSceneEvent({
        id: String(evt.id ?? Date.now()),
        type: (String(evt.type ?? "info")) as SceneEvent["type"],
        message: String(evt.message ?? ""),
        timestamp: String(evt.timestamp ?? new Date().toISOString()),
        severity: evt.severity as SceneEvent["severity"],
      });
    }
  });

  useEffect(() => {
    getSceneEvents()
      .then((events) => setSceneEvents(events))
      .catch(() => {});
  }, [setSceneEvents]);

  const filtered =
    filter === "all"
      ? sceneEvents
      : sceneEvents.filter((e) => e.type === filter);

  const filters: FilterType[] = [
    "all",
    "detection",
    "alert",
    "warning",
    "info",
    "system",
  ];

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-700/50 bg-slate-800 shadow-lg">
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
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h2 className="text-sm font-semibold text-slate-200">Events</h2>
        </div>
        <span className="rounded-md bg-slate-700/50 px-2 py-0.5 text-xs text-slate-400">
          {filtered.length} events
        </span>
      </div>

      {/* Filters */}
      <div className="flex gap-1 overflow-x-auto border-b border-slate-700/30 px-4 py-2">
        {filters.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
              filter === f
                ? "bg-indigo-600/20 text-indigo-300"
                : "text-slate-400 hover:bg-slate-700/50 hover:text-slate-300"
            }`}
          >
            {f === "all" ? "All" : f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Event List */}
      <div className="flex-1 overflow-y-auto p-3">
        {filtered.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 py-8">
            <svg
              className="h-8 w-8 text-slate-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-xs text-slate-500">No events yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map((event) => (
              <div
                key={event.id}
                className={`rounded-lg border px-3 py-2.5 transition-colors hover:bg-slate-700/30 ${
                  EVENT_COLORS[event.type] ?? EVENT_COLORS.info
                }`}
              >
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-sm">
                    {EVENT_ICONS[event.type] ?? "📌"}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm leading-snug text-slate-200">
                      {event.message}
                    </p>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="text-[10px] text-slate-500">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                      {event.severity && (
                        <span className="flex items-center gap-1">
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              SEVERITY_DOT[event.severity] ?? SEVERITY_DOT.low
                            }`}
                          />
                          <span className="text-[10px] text-slate-500">
                            {event.severity}
                          </span>
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
