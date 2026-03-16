import { Card, CardTitle } from '@/components/common/Card';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { PageHeader } from '@/components/layout/PageHeader';
import { useAnalyticsExportUrl, useAnalyticsLive } from '@/services/api/hooks';
import { formatNumber, formatTime } from '@/utils/format';

export const AnalyticsPage = () => {
  const analytics = useAnalyticsLive();
  const exportUrl = useAnalyticsExportUrl();

  return (
    <div className="space-y-4">
      <PageHeader
        title="Analytics"
        subtitle="Live counts, rolling averages, and event export"
        actions={<a href={exportUrl} className="text-sm text-indigo-300 hover:underline">Export CSV</a>}
      />

      {analytics.error ? <ErrorState>{(analytics.error as Error).message}</ErrorState> : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="space-y-3">
          <CardTitle>Current counts</CardTitle>
          {Object.keys(analytics.data?.current_counts ?? {}).length === 0 ? (
            <EmptyState title="No detections" description="Counts will appear when tracked objects are present." />
          ) : (
            <ul className="space-y-1 text-sm text-slate-300">
              {Object.entries(analytics.data?.current_counts ?? {}).map(([name, count]) => (
                <li key={name} className="flex justify-between rounded bg-slate-800/60 px-2 py-1">
                  <span>{name}</span>
                  <span>{formatNumber(count)}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card className="space-y-3">
          <CardTitle>Rolling averages</CardTitle>
          <ul className="space-y-1 text-sm text-slate-300">
            {Object.entries(analytics.data?.rolling_avg_counts ?? {}).map(([name, count]) => (
              <li key={name} className="flex justify-between rounded bg-slate-800/60 px-2 py-1">
                <span>{name}</span>
                <span>{count}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card className="space-y-3">
        <CardTitle>Recent entry/exit events</CardTitle>
        {analytics.data?.recent_events.length ? (
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="text-left text-slate-400">
                <tr>
                  <th className="py-2">Time</th>
                  <th>Type</th>
                  <th>Track ID</th>
                  <th>Class</th>
                </tr>
              </thead>
              <tbody>
                {analytics.data.recent_events.map((event, index) => (
                  <tr key={`${event.track_id}-${index}`} className="border-t border-slate-800">
                    <td className="py-2">{formatTime(event.timestamp)}</td>
                    <td>{event.event_type}</td>
                    <td>{event.track_id}</td>
                    <td>{event.class_name}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState title="No events yet" />
        )}
      </Card>
    </div>
  );
};
