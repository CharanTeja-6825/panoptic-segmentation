import { useEffect, useMemo } from "react";
import { useStore } from "../store/useStore";
import { getBenchmark, getLiveAnalytics } from "../services/api";
import { summarizeObjectCounts } from "../lib/mappers";

const CARD_ICONS: Record<string, string> = {
  Objects: "M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15",
  FPS: "M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z",
  Latency: "M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z",
  Classes: "M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z",
};

export default function StatsPanel() {
  const analytics = useStore((s) => s.analytics);
  const sceneObjects = useStore((s) => s.sceneObjects);
  const setAnalytics = useStore((s) => s.setAnalytics);
  const addSceneEvent = useStore((s) => s.addSceneEvent);

  useEffect(() => {
    let mounted = true;

    const fetchAnalytics = async () => {
      const [live, bench] = await Promise.all([
        getLiveAnalytics().catch(() => null),
        getBenchmark().catch(() => null),
      ]);
      if (!mounted) return;

      if (!live && !bench) {
        addSceneEvent({
          id: `analytics-${Date.now()}`,
          type: "system",
          message: "Analytics endpoint is temporarily unavailable.",
          timestamp: new Date().toISOString(),
          severity: "medium",
        });
        return;
      }

      const objectCounts = summarizeObjectCounts(sceneObjects);
      setAnalytics({
        totalDetections: sceneObjects.length,
        objectCounts,
        fps: Number(bench?.fps ?? analytics.fps ?? 0),
        latency: Number(bench?.latency ?? analytics.latency ?? 0),
        ...(live as Record<string, unknown> | null),
      });
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [sceneObjects, setAnalytics, analytics.fps, analytics.latency, addSceneEvent]);

  const objectEntries = useMemo(
    () => Object.entries(analytics.objectCounts).sort(([, a], [, b]) => b - a),
    [analytics.objectCounts]
  );
  const maxCount = Math.max(...objectEntries.map(([, c]) => c), 1);
  const avgConfidence =
    sceneObjects.length > 0
      ? (sceneObjects.reduce((sum, obj) => sum + obj.confidence, 0) /
          sceneObjects.length) *
        100
      : 0;

  const cards = [
    { label: "Objects", value: analytics.totalDetections, hint: "Live detections" },
    { label: "FPS", value: analytics.fps.toFixed(1), hint: "Inference speed" },
    { label: "Latency", value: `${analytics.latency.toFixed(0)}ms`, hint: "Pipeline delay" },
    { label: "Classes", value: objectEntries.length, hint: "Unique labels" },
  ];

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Analytics Overview</h2>
          <p className="mt-0.5 text-xs" style={{ color: "var(--color-text-muted)" }}>
            Metrics &amp; object distribution
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-2 gap-3">
          {cards.map((card) => (
            <div key={card.label} className="metric-card">
              <div className="mb-2 flex items-center gap-2">
                <div
                  className="flex h-7 w-7 items-center justify-center rounded-lg"
                  style={{ background: "var(--color-accent-glow)" }}
                >
                  <svg className="h-3.5 w-3.5" style={{ color: "var(--color-accent-hover)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={CARD_ICONS[card.label]} />
                  </svg>
                </div>
                <span className="text-[11px] font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
                  {card.label}
                </span>
              </div>
              <p className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
                {card.value}
              </p>
              <p className="mt-0.5 text-[11px]" style={{ color: "var(--color-text-muted)" }}>
                {card.hint}
              </p>
            </div>
          ))}
        </div>

        {/* Distribution */}
        <div className="mt-4 rounded-xl p-4" style={{ background: "var(--color-bg-secondary)", border: "1px solid var(--color-border)" }}>
          <h3 className="mb-3 text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
            Distribution by class
          </h3>
          {objectEntries.length === 0 ? (
            <p className="py-3 text-center text-sm" style={{ color: "var(--color-text-muted)" }}>
              Waiting for detections…
            </p>
          ) : (
            <div className="space-y-3">
              {objectEntries.map(([label, count]) => (
                <div key={label}>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span style={{ color: "var(--color-text-secondary)" }}>{label}</span>
                    <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>{count}</span>
                  </div>
                  <div className="progress-bar" style={{ height: "6px" }}>
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${(count / maxCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Confidence */}
        <div className="mt-4 rounded-xl p-4" style={{ background: "var(--color-bg-secondary)", border: "1px solid var(--color-border)" }}>
          <h3 className="mb-3 text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
            Confidence
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span style={{ color: "var(--color-text-secondary)" }}>Average confidence</span>
              <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>
                {sceneObjects.length ? `${avgConfidence.toFixed(1)}%` : "—"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span style={{ color: "var(--color-text-secondary)" }}>Total tracked objects</span>
              <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>
                {analytics.totalDetections}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
