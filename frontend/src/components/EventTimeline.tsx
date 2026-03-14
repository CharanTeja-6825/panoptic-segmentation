import { useEffect, useMemo, useState } from "react";
import { useStore, SceneEvent } from "../store/useStore";
import { getSceneEvents } from "../services/api";
import { eventSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";
import { normalizeSceneEvent } from "../lib/mappers";

type FilterType = "all" | SceneEvent["type"];

const EVENT_STYLES: Record<SceneEvent["type"], { border: string; bg: string; text: string; dot: string }> = {
  detection: { border: "rgba(99,102,241,0.25)", bg: "rgba(99,102,241,0.08)", text: "#c7d2fe", dot: "#818cf8" },
  alert:     { border: "rgba(248,113,113,0.25)", bg: "rgba(248,113,113,0.08)", text: "#fecaca", dot: "#f87171" },
  info:      { border: "rgba(56,189,248,0.25)", bg: "rgba(56,189,248,0.08)", text: "#bae6fd", dot: "#38bdf8" },
  warning:   { border: "rgba(251,191,36,0.25)", bg: "rgba(251,191,36,0.08)", text: "#fef08a", dot: "#fbbf24" },
  system:    { border: "var(--color-border)", bg: "var(--color-bg-secondary)", text: "var(--color-text-secondary)", dot: "#64748b" },
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

  const filters: FilterType[] = ["all", "detection", "alert", "warning", "info", "system"];

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Event Timeline</h2>
          <p className="mt-0.5 text-xs" style={{ color: "var(--color-text-muted)" }}>
            Live scene events &amp; behavior changes
          </p>
        </div>
        <span className="status-pill status-pill-muted">{filtered.length} events</span>
      </header>

      {/* Filter bar */}
      <div className="px-4 py-2.5" style={{ borderBottom: "1px solid var(--color-border)" }}>
        <div className="flex flex-wrap gap-1.5">
          {filters.map((item) => (
            <button
              key={item}
              onClick={() => setFilter(item)}
              className={filter === item ? "filter-chip filter-chip-active" : "filter-chip"}
            >
              {item === "all" ? "All" : item.charAt(0).toUpperCase() + item.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Events list */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {filtered.length === 0 ? (
          <div className="empty-state" style={{ padding: "2rem" }}>
            <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
              No events available for this filter.
            </p>
          </div>
        ) : (
          <ol className="space-y-2">
            {filtered.map((event) => {
              const style = EVENT_STYLES[event.type];
              return (
                <li
                  key={event.id}
                  className="fade-in rounded-xl px-3.5 py-3"
                  style={{
                    background: style.bg,
                    border: `1px solid ${style.border}`,
                    color: style.text,
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <p className="min-w-0 flex-1 break-words text-sm leading-snug">
                      {event.message}
                    </p>
                    <span className="shrink-0 text-[11px]" style={{ color: "var(--color-text-muted)" }}>
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-[11px]">
                    <span className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5" style={{ background: "rgba(0,0,0,0.2)" }}>
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: style.dot }} />
                      {event.type}
                    </span>
                    {event.severity && (
                      <span className="rounded-md px-1.5 py-0.5 uppercase tracking-wide" style={{ background: "rgba(0,0,0,0.2)" }}>
                        {event.severity}
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}
