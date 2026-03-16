import { Card, CardTitle } from '@/components/common/Card';
import type { VideoAnalysisPayload } from '@/types/api';
import { formatNumber } from '@/utils/format';

interface AnalysisSummaryProps {
  analysis: VideoAnalysisPayload;
}

export const AnalysisSummary = ({ analysis }: AnalysisSummaryProps) => (
  <div className="grid gap-4 lg:grid-cols-2">
    <Card className="space-y-3">
      <CardTitle>Analysis summary</CardTitle>
      <ul className="space-y-1 text-sm text-slate-300">
        <li>Unique tracks: {formatNumber(analysis.summary.total_unique_tracks)}</li>
        <li>Events: {formatNumber(analysis.summary.events_count)}</li>
        <li>Keyframes: {formatNumber(analysis.summary.keyframes_count)}</li>
        <li>Processed frames: {formatNumber(analysis.video.frames_processed)}</li>
      </ul>
    </Card>

    <Card className="space-y-3">
      <CardTitle>Keyframes</CardTitle>
      {analysis.keyframes.length === 0 ? (
        <p className="text-sm text-slate-400">No keyframes generated.</p>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          {analysis.keyframes.slice(0, 6).map((item) => (
            <a key={item.file_name} href={item.url} target="_blank" rel="noreferrer" className="overflow-hidden rounded border border-slate-800">
              <img src={item.url} alt={`Frame ${item.frame_index}`} className="h-24 w-full object-cover" />
            </a>
          ))}
        </div>
      )}
    </Card>
  </div>
);
