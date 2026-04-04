import { useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { Alert, AlertDescription, AlertIcons } from '@/components/common/Alert';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { Input } from '@/components/common/Input';
import { Select } from '@/components/common/Select';
import { MJPEGViewer } from '@/components/features/MJPEGViewer';
import { PageHeader } from '@/components/layout/PageHeader';
import { queryKeys } from '@/constants/queryKeys';
import {
  useCameraStatus,
  useLlmVisionModels,
  useMjpegUrl,
  useMultiCameraList,
  useSceneSummary,
  useStartCamera,
  useStartMultiCamera,
  useStopCamera,
  useStopMultiCamera,
  useToggleDepth,
  useToggleHeatmap,
} from '@/services/api/hooks';
import { useCameraStore } from '@/store/cameraStore';

export const CameraPage = () => {
  const queryClient = useQueryClient();
  const status = useCameraStatus();
  const streamList = useMultiCameraList();
  const start = useStartCamera();
  const stop = useStopCamera();
  const toggleHeatmap = useToggleHeatmap();
  const toggleDepth = useToggleDepth();
  const startMulti = useStartMultiCamera();
  const stopMulti = useStopMultiCamera();
  const mjpegUrl = useMjpegUrl();
  const sceneSummary = useSceneSummary();
  const llmVisionModels = useLlmVisionModels();

  const [selectedVisionModel, setSelectedVisionModel] = useState('');

  const { cameraIndex, cameraLabel, setCameraIndex, setCameraLabel } = useCameraStore();

  const visionModels = useMemo(() => llmVisionModels.data?.models ?? [], [llmVisionModels.data?.models]);
  const defaultVisionModel = llmVisionModels.data?.default ?? 'llava-phi3';

  const refreshCameras = async () => {
    await queryClient.invalidateQueries({ queryKey: queryKeys.cameraStatus });
    await queryClient.invalidateQueries({ queryKey: queryKeys.cameraList });
  };

  return (
    <div className="space-y-4">
      <PageHeader 
        title="Camera" 
        subtitle="Live MJPEG stream with AI-powered scene insights" 
      />

      {status.error && (
        <ErrorState>{(status.error as Error).message}</ErrorState>
      )}

      <div className="grid gap-4 xl:grid-cols-[2fr,1fr]">
        {/* Camera view */}
        <div className="space-y-4">
          <MJPEGViewer src={mjpegUrl} />
          
          {/* Scene insights panel */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Scene Insights</CardTitle>
                <Badge variant={status.data?.running ? 'success' : 'secondary'} dot>
                  {status.data?.running ? 'Live' : 'Offline'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Quick stats */}
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                  <p className="text-2xl font-bold text-indigo-400">
                    {sceneSummary.data?.active_objects ?? 0}
                  </p>
                  <p className="text-xs text-slate-400">Active Objects</p>
                </div>
                <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                  <p className="text-2xl font-bold text-emerald-400">
                    {sceneSummary.data?.total_unique_objects_seen ?? 0}
                  </p>
                  <p className="text-xs text-slate-400">Total Tracked</p>
                </div>
                <div className="rounded-lg bg-slate-800/50 p-3 text-center">
                  <p className="text-2xl font-bold text-amber-400">
                    {sceneSummary.data?.recent_events?.length ?? 0}
                  </p>
                  <p className="text-xs text-slate-400">Recent Events</p>
                </div>
              </div>

              {/* Object class breakdown */}
              {sceneSummary.data?.counts_by_class && 
               Object.keys(sceneSummary.data.counts_by_class).length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-300">Objects by Class</h4>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(sceneSummary.data.counts_by_class).map(([cls, count]) => (
                      <Badge key={cls} variant="outline">
                        {cls}: {count}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Latest AI snapshot */}
              {sceneSummary.data?.latest_snapshot?.summary_text && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-300">AI Analysis</h4>
                  <p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200">
                    {sceneSummary.data.latest_snapshot.summary_text}
                  </p>
                </div>
              )}

              {/* Recent events */}
              {sceneSummary.data?.recent_events && sceneSummary.data.recent_events.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-300">Recent Events</h4>
                  <div className="max-h-32 space-y-1 overflow-y-auto scrollbar-thin">
                    {sceneSummary.data.recent_events.slice(0, 5).map((event, idx) => (
                      <div 
                        key={idx} 
                        className="flex items-center gap-2 rounded bg-slate-800/50 px-2 py-1 text-xs"
                      >
                        <Badge 
                          size="sm"
                          variant={event.event_type === 'entry' ? 'success' : 
                                   event.event_type === 'exit' ? 'warning' : 'secondary'}
                        >
                          {event.event_type}
                        </Badge>
                        <span className="text-slate-300">{event.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Controls panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Camera Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Vision model selector */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Vision Model</label>
                <Select
                  value={selectedVisionModel}
                  onChange={(e) => setSelectedVisionModel(e.target.value)}
                >
                  <option value="">Default ({defaultVisionModel})</option>
                  {visionModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </Select>
                <p className="text-xs text-slate-500">
                  Used for AI-powered scene analysis
                </p>
              </div>

              {/* Control buttons */}
              <div className="grid grid-cols-2 gap-2">
                <Button 
                  onClick={async () => { await start.mutateAsync(); await refreshCameras(); }}
                  loading={start.isPending}
                  disabled={status.data?.running}
                >
                  Start
                </Button>
                <Button 
                  variant="destructive" 
                  onClick={async () => { await stop.mutateAsync(); await refreshCameras(); }}
                  loading={stop.isPending}
                  disabled={!status.data?.running}
                >
                  Stop
                </Button>
                <Button 
                  variant="secondary" 
                  onClick={async () => { await toggleHeatmap.mutateAsync(); await refreshCameras(); }}
                  loading={toggleHeatmap.isPending}
                >
                  {status.data?.heatmap ? 'Disable' : 'Enable'} Heatmap
                </Button>
                <Button 
                  variant="secondary" 
                  onClick={async () => { await toggleDepth.mutateAsync(); await refreshCameras(); }}
                  loading={toggleDepth.isPending}
                >
                  Toggle Depth
                </Button>
              </div>

              {/* Status info */}
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-slate-400">Running</dt>
                  <dd>
                    <Badge variant={status.data?.running ? 'success' : 'secondary'}>
                      {status.data?.running ? 'Yes' : 'No'}
                    </Badge>
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-400">FPS</dt>
                  <dd className="text-slate-200">{status.data?.fps ?? 0}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-400">Heatmap</dt>
                  <dd>
                    <Badge variant={status.data?.heatmap ? 'info' : 'secondary'}>
                      {status.data?.heatmap ? 'On' : 'Off'}
                    </Badge>
                  </dd>
                </div>
              </dl>
            </CardContent>
          </Card>

          {/* Multi-camera start */}
          <Card>
            <CardHeader>
              <CardTitle>Add Camera Stream</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                type="number"
                min={0}
                value={cameraIndex}
                onChange={(event) => setCameraIndex(Number(event.target.value))}
                placeholder="Camera index"
                label="Camera Index"
              />
              <Input 
                value={cameraLabel} 
                onChange={(event) => setCameraLabel(event.target.value)} 
                placeholder="Label (e.g., Front Door)"
                label="Label"
              />
              <Button
                className="w-full"
                onClick={async () => {
                  await startMulti.mutateAsync({ camera_index: cameraIndex, label: cameraLabel });
                  await refreshCameras();
                }}
                loading={startMulti.isPending}
              >
                Start Stream
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Multi-camera streams table */}
      <Card>
        <CardHeader>
          <CardTitle>Active Camera Streams</CardTitle>
        </CardHeader>
        <CardContent>
          {streamList.data?.streams.length ? (
            <div className="overflow-auto">
              <table className="min-w-full text-sm">
                <thead className="text-left text-slate-400">
                  <tr className="border-b border-slate-700">
                    <th className="pb-2">ID</th>
                    <th>Camera</th>
                    <th>Label</th>
                    <th>FPS</th>
                    <th>Objects</th>
                    <th className="text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {streamList.data.streams.map((stream) => (
                    <tr key={stream.stream_id} className="border-t border-slate-800">
                      <td className="py-3 font-mono text-xs text-slate-300">
                        {stream.stream_id}
                      </td>
                      <td>{stream.camera_index}</td>
                      <td>{stream.label}</td>
                      <td>
                        <Badge variant="secondary">{stream.fps} fps</Badge>
                      </td>
                      <td>
                        <Badge variant="info">{stream.object_count}</Badge>
                      </td>
                      <td className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={async () => {
                            await stopMulti.mutateAsync({ stream_id: stream.stream_id });
                            await refreshCameras();
                          }}
                          loading={stopMulti.isPending}
                        >
                          Stop
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState 
              title="No extra streams" 
              description="Start a camera stream to see it listed here." 
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
};
