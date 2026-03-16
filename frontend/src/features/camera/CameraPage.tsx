import { useQueryClient } from '@tanstack/react-query';

import { Button } from '@/components/common/Button';
import { Card, CardTitle } from '@/components/common/Card';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { Input } from '@/components/common/Input';
import { MJPEGViewer } from '@/components/features/MJPEGViewer';
import { PageHeader } from '@/components/layout/PageHeader';
import { queryKeys } from '@/constants/queryKeys';
import {
  useCameraStatus,
  useMjpegUrl,
  useMultiCameraList,
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

  const { cameraIndex, cameraLabel, setCameraIndex, setCameraLabel } = useCameraStore();

  const refreshCameras = async () => {
    await queryClient.invalidateQueries({ queryKey: queryKeys.cameraStatus });
    await queryClient.invalidateQueries({ queryKey: queryKeys.cameraList });
  };

  return (
    <div className="space-y-4">
      <PageHeader title="Camera" subtitle="Live MJPEG stream and camera controls" />

      {status.error ? <ErrorState>{(status.error as Error).message}</ErrorState> : null}

      <div className="grid gap-4 xl:grid-cols-[2fr,1fr]">
        <MJPEGViewer src={mjpegUrl} />

        <Card className="space-y-3">
          <CardTitle>Primary camera controls</CardTitle>
          <div className="grid grid-cols-2 gap-2">
            <Button onClick={async () => { await start.mutateAsync(); await refreshCameras(); }}>Start</Button>
            <Button variant="danger" onClick={async () => { await stop.mutateAsync(); await refreshCameras(); }}>Stop</Button>
            <Button variant="secondary" onClick={async () => { await toggleHeatmap.mutateAsync(); await refreshCameras(); }}>
              Toggle heatmap
            </Button>
            <Button variant="secondary" onClick={async () => { await toggleDepth.mutateAsync(); await refreshCameras(); }}>
              Toggle depth
            </Button>
          </div>
          <dl className="space-y-1 text-sm text-slate-300">
            <div className="flex justify-between"><dt>Running</dt><dd>{String(status.data?.running ?? false)}</dd></div>
            <div className="flex justify-between"><dt>FPS</dt><dd>{status.data?.fps ?? 0}</dd></div>
            <div className="flex justify-between"><dt>Heatmap</dt><dd>{String(status.data?.heatmap ?? false)}</dd></div>
          </dl>
        </Card>
      </div>

      <Card className="space-y-3">
        <CardTitle>Start additional camera stream</CardTitle>
        <div className="grid gap-2 md:grid-cols-3">
          <Input
            type="number"
            min={0}
            value={cameraIndex}
            onChange={(event) => setCameraIndex(Number(event.target.value))}
            aria-label="Camera index"
          />
          <Input value={cameraLabel} onChange={(event) => setCameraLabel(event.target.value)} placeholder="Label" aria-label="Camera label" />
          <Button
            onClick={async () => {
              await startMulti.mutateAsync({ camera_index: cameraIndex, label: cameraLabel });
              await refreshCameras();
            }}
          >
            Start stream
          </Button>
        </div>
      </Card>

      <Card className="space-y-3">
        <CardTitle>Active multi-camera streams</CardTitle>
        {streamList.data?.streams.length ? (
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="text-left text-slate-400">
                <tr>
                  <th className="py-2">ID</th>
                  <th>Camera</th>
                  <th>Label</th>
                  <th>FPS</th>
                  <th>Objects</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {streamList.data.streams.map((stream) => (
                  <tr key={stream.stream_id} className="border-t border-slate-800">
                    <td className="py-2 font-mono text-xs text-slate-300">{stream.stream_id}</td>
                    <td>{stream.camera_index}</td>
                    <td>{stream.label}</td>
                    <td>{stream.fps}</td>
                    <td>{stream.object_count}</td>
                    <td>
                      <Button
                        variant="ghost"
                        onClick={async () => {
                          await stopMulti.mutateAsync({ stream_id: stream.stream_id });
                          await refreshCameras();
                        }}
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
          <EmptyState title="No extra streams" description="Start a camera stream to see it listed here." />
        )}
      </Card>
    </div>
  );
};
