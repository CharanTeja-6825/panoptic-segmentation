import { Card, CardTitle } from '@/components/common/Card';
import { ErrorState } from '@/components/common/ErrorState';
import { Skeleton } from '@/components/common/Skeleton';
import { StatTile } from '@/components/common/StatTile';
import { EventFeed } from '@/components/features/EventFeed';
import { PageHeader } from '@/components/layout/PageHeader';
import { useAnalyticsLive, useBenchmark, useHealth, useSceneSummary } from '@/services/api/hooks';
import { useWsStore } from '@/store/wsStore';
import { formatNumber } from '@/utils/format';

export const DashboardPage = () => {
  const health = useHealth();
  const benchmark = useBenchmark();
  const analytics = useAnalyticsLive();
  const sceneSummary = useSceneSummary();
  const wsEvents = useWsStore((state) => state.latestEvents);

  return (
    <div className="space-y-4">
      <PageHeader title="Dashboard" subtitle="Realtime operational overview" />

      {(health.error || analytics.error || sceneSummary.error) ? (
        <ErrorState>Failed to load one or more dashboard panels.</ErrorState>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Health" value={health.data?.status ?? (health.isLoading ? '...' : 'unknown')} helper={health.data?.device} />
        <StatTile label="Scene active objects" value={sceneSummary.data?.active_objects ?? 0} helper={`Total seen: ${sceneSummary.data?.total_unique_objects_seen ?? 0}`} />
        <StatTile label="Live FPS" value={analytics.data?.fps ?? 0} helper={`Frames measured: ${benchmark.data?.frames_measured ?? 0}`} />
        <StatTile label="Current tracked" value={analytics.data?.total_objects ?? 0} helper={`Total frames: ${analytics.data?.total_frames ?? 0}`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="space-y-3">
          <CardTitle>Health + benchmark</CardTitle>
          {health.isLoading || benchmark.isLoading ? (
            <Skeleton className="h-24" />
          ) : (
            <dl className="grid gap-2 text-sm text-slate-300 sm:grid-cols-2">
              <div>
                <dt className="text-slate-400">GPU</dt>
                <dd>{health.data?.gpu ?? 'N/A'}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Model size</dt>
                <dd>{health.data?.model_size}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Average FPS</dt>
                <dd>{benchmark.data?.avg_fps ?? 0}</dd>
              </div>
              <div>
                <dt className="text-slate-400">CPU load (1m)</dt>
                <dd>{benchmark.data?.cpu_load_1m ?? 0}</dd>
              </div>
            </dl>
          )}
        </Card>

        <Card className="space-y-3">
          <CardTitle>Current class counts</CardTitle>
          {analytics.isLoading ? (
            <Skeleton className="h-24" />
          ) : (
            <ul className="space-y-1 text-sm text-slate-300">
              {Object.entries(analytics.data?.current_counts ?? {}).map(([label, count]) => (
                <li key={label} className="flex justify-between rounded bg-slate-800/60 px-2 py-1">
                  <span>{label}</span>
                  <span>{formatNumber(count)}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card className="space-y-3">
        <CardTitle>Live event feed (WebSocket)</CardTitle>
        <EventFeed
          items={wsEvents.slice(0, 15).map((item) => ({
            id: item.id,
            message: item.message,
            timestamp: new Date(item.timestamp).toLocaleTimeString(),
            severity: item.severity,
          }))}
        />
      </Card>
    </div>
  );
};
