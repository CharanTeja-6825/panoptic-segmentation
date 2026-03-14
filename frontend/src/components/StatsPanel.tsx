import { useEffect, useMemo } from "react";
import { useStore } from "../store/useStore";
import { getBenchmark, getLiveAnalytics } from "../services/api";
import { summarizeObjectCounts } from "../lib/mappers";

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
    {
      label: "Latency",
      value: `${analytics.latency.toFixed(0)}ms`,
      hint: "Pipeline delay",
    },
    { label: "Classes", value: objectEntries.length, hint: "Unique labels" },
  ];

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Analytics Overview</h2>
          <p className="mt-0.5 text-xs text-slate-400">
            Operational metrics and object distribution
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-2 gap-2">
          {cards.map((card) => (
            <article
              key={card.label}
              className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2.5"
            >
              <p className="text-lg font-semibold text-slate-100">{card.value}</p>
              <p className="text-[11px] uppercase tracking-wider text-slate-400">
                {card.label}
              </p>
              <p className="mt-1 text-[11px] text-slate-500">{card.hint}</p>
            </article>
          ))}
        </div>

        <section className="mt-4 rounded-xl border border-slate-700 bg-slate-900 p-3">
          <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Distribution by class
          </h3>
          {objectEntries.length === 0 ? (
            <p className="py-4 text-center text-sm text-slate-500">
              Waiting for detections...
            </p>
          ) : (
            <div className="mt-3 space-y-2.5">
              {objectEntries.map(([label, count]) => (
                <div key={label}>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="text-slate-300">{label}</span>
                    <span className="font-semibold text-slate-100">{count}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-indigo-500"
                      style={{ width: `${(count / maxCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="mt-4 rounded-xl border border-slate-700 bg-slate-900 p-3 text-xs">
          <h3 className="mb-2 uppercase tracking-wider text-slate-400">Confidence</h3>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Average confidence</span>
            <span className="font-semibold text-slate-100">
              {sceneObjects.length ? `${avgConfidence.toFixed(1)}%` : "—"}
            </span>
          </div>
          <div className="mt-1.5 flex items-center justify-between">
            <span className="text-slate-400">Total tracked objects</span>
            <span className="font-semibold text-slate-100">
              {analytics.totalDetections}
            </span>
          </div>
        </section>
      </div>
    </section>
  );
}
