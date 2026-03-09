import { useEffect } from "react";
import { useStore } from "../store/useStore";
import { getLiveAnalytics, getBenchmark } from "../services/api";

export default function StatsPanel() {
  const analytics = useStore((s) => s.analytics);
  const setAnalytics = useStore((s) => s.setAnalytics);
  const sceneObjects = useStore((s) => s.sceneObjects);

  useEffect(() => {
    let mounted = true;

    const fetchAnalytics = async () => {
      try {
        const [live, bench] = await Promise.all([
          getLiveAnalytics().catch(() => null),
          getBenchmark().catch(() => null),
        ]);
        if (!mounted) return;

        const objectCounts = sceneObjects.reduce<Record<string, number>>(
          (acc, obj) => {
            acc[obj.label] = (acc[obj.label] ?? 0) + 1;
            return acc;
          },
          {}
        );

        setAnalytics({
          totalDetections: sceneObjects.length,
          objectCounts,
          fps: bench?.fps ?? analytics.fps,
          latency: bench?.latency ?? analytics.latency,
          ...(live as Record<string, unknown> | null),
        });
      } catch {
        // Silently handle errors
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [sceneObjects, setAnalytics, analytics.fps, analytics.latency]);

  const objectEntries = Object.entries(analytics.objectCounts);
  const maxCount = Math.max(...objectEntries.map(([, c]) => c), 1);

  const statCards = [
    {
      label: "Objects",
      value: analytics.totalDetections,
      color: "text-indigo-400",
      bg: "bg-indigo-500/10",
    },
    {
      label: "FPS",
      value: analytics.fps.toFixed(1),
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
    },
    {
      label: "Latency",
      value: `${analytics.latency.toFixed(0)}ms`,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
    },
    {
      label: "Classes",
      value: objectEntries.length,
      color: "text-sky-400",
      bg: "bg-sky-500/10",
    },
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
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <h2 className="text-sm font-semibold text-slate-200">Analytics</h2>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {/* Stat Cards */}
        <div className="mb-4 grid grid-cols-2 gap-2">
          {statCards.map((stat) => (
            <div
              key={stat.label}
              className={`rounded-lg ${stat.bg} p-3 text-center`}
            >
              <p className={`text-xl font-bold ${stat.color}`}>{stat.value}</p>
              <p className="mt-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400">
                {stat.label}
              </p>
            </div>
          ))}
        </div>

        {/* Object Distribution */}
        <div>
          <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-400">
            Object Distribution
          </h3>
          {objectEntries.length === 0 ? (
            <p className="py-4 text-center text-xs text-slate-500">
              No detections yet
            </p>
          ) : (
            <div className="space-y-2.5">
              {objectEntries
                .sort(([, a], [, b]) => b - a)
                .map(([label, count]) => (
                  <div key={label}>
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-xs text-slate-300">{label}</span>
                      <span className="text-xs font-semibold text-slate-200">
                        {count}
                      </span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-slate-700">
                      <div
                        className="h-full rounded-full bg-indigo-500 transition-all duration-500"
                        style={{
                          width: `${(count / maxCount) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>

        {/* Activity Summary */}
        <div className="mt-4 rounded-lg border border-slate-700/30 bg-slate-700/20 p-3">
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">
            Activity Summary
          </h3>
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Total detections</span>
            <span className="font-semibold text-slate-200">
              {analytics.totalDetections}
            </span>
          </div>
          <div className="mt-1.5 flex items-center justify-between text-xs">
            <span className="text-slate-400">Unique classes</span>
            <span className="font-semibold text-slate-200">
              {objectEntries.length}
            </span>
          </div>
          <div className="mt-1.5 flex items-center justify-between text-xs">
            <span className="text-slate-400">Avg confidence</span>
            <span className="font-semibold text-slate-200">
              {sceneObjects.length > 0
                ? `${(
                    (sceneObjects.reduce((s, o) => s + o.confidence, 0) /
                      sceneObjects.length) *
                    100
                  ).toFixed(1)}%`
                : "—"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
