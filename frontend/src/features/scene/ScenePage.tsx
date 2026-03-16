import { useMemo, useState } from 'react';

import { Card, CardTitle } from '@/components/common/Card';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { Input } from '@/components/common/Input';
import { EventFeed } from '@/components/features/EventFeed';
import { PageHeader } from '@/components/layout/PageHeader';
import {
  useSceneCommentary,
  useSceneEvents,
  useSceneHistory,
  useSceneState,
  useSceneSummary,
  useSceneTimeSummary,
} from '@/services/api/hooks';
import { useWsStore } from '@/store/wsStore';
import { formatTime } from '@/utils/format';

export const ScenePage = () => {
  const [classFilter, setClassFilter] = useState('');
  const [minutesText, setMinutesText] = useState('60');

  const minutes = Number(minutesText) > 0 ? Number(minutesText) : 60;

  const state = useSceneState();
  const events = useSceneEvents();
  const summary = useSceneSummary();
  const history = useSceneHistory(classFilter, 100);
  const timeSummary = useSceneTimeSummary(minutes);
  const commentary = useSceneCommentary();

  const wsEvents = useWsStore((store) => store.latestEvents);

  const mergedFeed = useMemo(
    () => [
      ...wsEvents.slice(0, 10).map((item) => ({
        id: item.id,
        message: item.message,
        timestamp: new Date(item.timestamp).toLocaleTimeString(),
        severity: item.severity,
      })),
      ...(events.data ?? []).slice(0, 10).map((item, index) => ({
        id: `${item.timestamp}-${index}`,
        message: item.description,
        timestamp: formatTime(item.timestamp),
        severity: item.event_type,
      })),
    ],
    [events.data, wsEvents],
  );

  return (
    <div className="space-y-4">
      <PageHeader title="Scene memory" subtitle="Current scene state, history, and commentary" />

      {(state.error || events.error || summary.error || history.error || commentary.error) ? (
        <ErrorState>One or more scene endpoints returned an error.</ErrorState>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="space-y-2">
          <CardTitle>Current scene state</CardTitle>
          <p className="text-sm text-slate-400">Active objects: {state.data?.active_count ?? 0}</p>
          <ul className="space-y-1 text-sm text-slate-300">
            {Object.entries(state.data?.counts_by_class ?? {}).map(([name, count]) => (
              <li key={name} className="flex justify-between rounded bg-slate-800/60 px-2 py-1">
                <span>{name}</span>
                <span>{count}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card className="space-y-2">
          <CardTitle>Time summary</CardTitle>
          <Input value={minutesText} onChange={(event) => setMinutesText(event.target.value)} type="number" min={1} />
          <dl className="space-y-1 text-sm text-slate-300">
            <div className="flex justify-between"><dt>Total events</dt><dd>{timeSummary.data?.total_events ?? 0}</dd></div>
            <div className="flex justify-between"><dt>Entries</dt><dd>{timeSummary.data?.entries ?? 0}</dd></div>
            <div className="flex justify-between"><dt>Exits</dt><dd>{timeSummary.data?.exits ?? 0}</dd></div>
          </dl>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="space-y-3">
          <CardTitle>Live events</CardTitle>
          <EventFeed items={mergedFeed.slice(0, 20)} />
        </Card>

        <Card className="space-y-3">
          <CardTitle>Commentary</CardTitle>
          {commentary.data?.commentary.length ? (
            <ul className="space-y-2 text-sm text-slate-300">
              {commentary.data.commentary.map((item, index) => (
                <li key={`${item.timestamp}-${index}`} className="rounded border border-slate-800 bg-slate-900 p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-500">{item.priority} · {item.type}</p>
                  <p className="mt-1">{item.text}</p>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No commentary yet" />
          )}
        </Card>
      </div>

      <Card className="space-y-3">
        <CardTitle>Object history</CardTitle>
        <div className="max-w-sm">
          <Input value={classFilter} onChange={(event) => setClassFilter(event.target.value)} placeholder="Filter by class (e.g. person)" />
        </div>
        {history.data?.length ? (
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="text-left text-slate-400">
                <tr>
                  <th className="py-2">Track</th>
                  <th>Class</th>
                  <th>Status</th>
                  <th>First seen</th>
                  <th>Last seen</th>
                </tr>
              </thead>
              <tbody>
                {history.data.map((item) => (
                  <tr key={`${item.track_id}-${item.last_seen}`} className="border-t border-slate-800">
                    <td className="py-2">{item.track_id}</td>
                    <td>{item.class_name}</td>
                    <td>{item.status}</td>
                    <td>{formatTime(item.first_seen)}</td>
                    <td>{formatTime(item.last_seen)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState title="No history records" />
        )}
      </Card>
    </div>
  );
};
