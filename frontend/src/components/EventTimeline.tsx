import { useEffect, useMemo, useState } from "react";
import { useStore, SceneEvent } from "../store/useStore";
import { getSceneEvents } from "../services/api";
import { eventSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";
import { normalizeSceneEvent } from "../lib/mappers";

type FilterType = "all" | SceneEvent["type"];

const EVENT_COLORS: Record<SceneEvent["type"], string> = {
  detection: "border-indigo-500/30 bg-indigo-500/10 text-indigo-100",
  alert: "border-rose-500/30 bg-rose-500/10 text-rose-100",
  info: "border-sky-500/30 bg-sky-500/10 text-sky-100",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
  system: "border-slate-500/30 bg-slate-500/10 text-slate-100",
};

export default function EventTimeline() {
  const sceneEvents = useStore((s) => s.sceneEvents);
  const addSceneEvent = useStore((s) => s.addSceneEvent);
  const setSceneEvents = useStore((s) => s.setSceneEvents);
  const [filter, setFilter] = useState<FilterType>("all");

  useWebSocket(eventSocket, "message", (data) => {
    const msg = data as Record<string, unknown>;
    if (msg.type === "event" && msg.event) {
      addSceneEvent(normalizeSceneEvent(msg.event as Record<string, unknown>));
    }
  });

  useEffect(() => {
    getSceneEvents()
      .then((events) => {
        const normalized = events.map((evt) =>
          normalizeSceneEvent(evt as Record<string, unknown>)
        );
        setSceneEvents(normalized);
      })
      .catch(() => {
        setSceneEvents([]);
      });
  }, [setSceneEvents]);

  const filtered = useMemo(() => {
    if (filter === "all") return sceneEvents;
    return sceneEvents.filter((event) => event.type === filter);
  }, [filter, sceneEvents]);

  const filters: FilterType[] = [
    "all",
    "detection",
    "alert",
    "warning",
    "info",
    "system",
  ];

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Event Timeline</h2>
          <p className="mt-0.5 text-xs text-slate-400">
            Live scene events from tracking and behavior changes
          </p>
        </div>
        <span className="status-pill status-pill-muted">{filtered.length} events</span>
      </header>

      <div className="border-b border-slate-700/60 px-4 py-2">
        <div className="flex flex-wrap gap-1.5">
          {filters.map((item) => (
            <button
              key={item}
              onClick={() => setFilter(item)}
              className={`rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
                filter === item
                  ? "border-indigo-500/40 bg-indigo-500/20 text-indigo-200"
                  : "border-slate-700 bg-slate-800 text-slate-300 hover:border-slate-600 hover:bg-slate-700"
              }`}
            >
              {item === "all"
                ? "All"
                : item.charAt(0).toUpperCase() + item.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {filtered.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">
            No events available for this filter.
          </div>
        ) : (
          <ol className="space-y-2">
            {filtered.map((event) => (
              <li
                key={event.id}
                className={`rounded-xl border px-3 py-2.5 ${EVENT_COLORS[event.type]}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="min-w-0 flex-1 break-words text-sm leading-snug">
                    {event.message}
                  </p>
                  <span className="shrink-0 text-[11px] text-slate-300">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-2 text-[11px]">
                  <span className="rounded bg-black/20 px-1.5 py-0.5 uppercase tracking-wide">
                    {event.type}
                  </span>
                  {event.severity && (
                    <span className="rounded bg-black/20 px-1.5 py-0.5 uppercase tracking-wide">
                      {event.severity}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}
