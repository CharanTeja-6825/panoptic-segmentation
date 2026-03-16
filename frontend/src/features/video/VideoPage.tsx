import { useState } from 'react';

import { Button } from '@/components/common/Button';
import { Card, CardTitle } from '@/components/common/Card';
import { ErrorState } from '@/components/common/ErrorState';
import { Input } from '@/components/common/Input';
import { AnalysisSummary } from '@/components/features/AnalysisSummary';
import { JobStatusBadge } from '@/components/features/JobStatusBadge';
import { UploadDropzone } from '@/components/features/UploadDropzone';
import { PageHeader } from '@/components/layout/PageHeader';
import {
  useDownloadUrl,
  useJobStatus,
  useProcessVideo,
  useUploadVideo,
  useVideoAnalysis,
  useVideoChat,
} from '@/services/api/hooks';
import { formatPercent } from '@/utils/format';

const MAX_BYTES = 500 * 1024 * 1024;

export const VideoPage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [question, setQuestion] = useState('');

  const upload = useUploadVideo();
  const processVideo = useProcessVideo();
  const status = useJobStatus(jobId);
  const analysis = useVideoAnalysis(jobId);
  const chat = useVideoChat(jobId);
  const downloadUrl = useDownloadUrl(jobId);
  const requestError = upload.error ?? processVideo.error ?? status.error ?? analysis.error ?? chat.error;

  const handleUpload = async () => {
    if (!selectedFile) {
      return;
    }

    if (selectedFile.size > MAX_BYTES) {
      alert('Selected file exceeds 500MB limit.');
      return;
    }

    const uploaded = await upload.mutateAsync(selectedFile);
    setJobId(uploaded.job_id);
    await processVideo.mutateAsync(uploaded.job_id);
  };

  return (
    <div className="space-y-4">
      <PageHeader title="Video processing" subtitle="Upload, process, analyze, and chat with processed videos" />

      {requestError ? (
        <ErrorState>{requestError.message}</ErrorState>
      ) : null}

      <Card className="space-y-3">
        <CardTitle>Upload and process</CardTitle>
        <UploadDropzone onSelect={setSelectedFile} />

        {selectedFile ? (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded border border-slate-800 bg-slate-900 p-3 text-sm">
            <div>
              <p className="font-medium text-slate-100">{selectedFile.name}</p>
              <p className="text-slate-400">{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
            </div>
            <Button disabled={upload.isPending || processVideo.isPending} onClick={handleUpload}>
              Upload & process
            </Button>
          </div>
        ) : null}

        {status.data ? (
          <div className="space-y-2 rounded border border-slate-800 bg-slate-900 p-3">
            <div className="flex items-center justify-between">
              <p className="text-sm text-slate-300">Job: <span className="font-mono text-xs">{status.data.job_id}</span></p>
              <JobStatusBadge status={status.data.status} />
            </div>
            <div className="h-2 rounded bg-slate-800">
              <div
                className="h-full rounded bg-indigo-500 transition-all"
                style={{ width: `${status.data.total_frames > 0 ? (status.data.progress / status.data.total_frames) * 100 : 0}%` }}
              />
            </div>
            <p className="text-xs text-slate-400">
              Frames: {status.data.progress}/{status.data.total_frames} | FPS: {status.data.fps} |
              {' '}Progress: {formatPercent(status.data.total_frames > 0 ? (status.data.progress / status.data.total_frames) * 100 : 0)}
            </p>
            {status.data.status === 'done' && downloadUrl ? (
              <a className="text-sm text-indigo-300 hover:underline" href={downloadUrl}>
                Download processed video
              </a>
            ) : null}
          </div>
        ) : null}
      </Card>

      {analysis.data?.analysis ? <AnalysisSummary analysis={analysis.data.analysis} /> : null}

      <Card className="space-y-3">
        <CardTitle>Video Q&A</CardTitle>
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about this processed video" />
          <Button
            disabled={!jobId || !question.trim() || chat.isPending}
            onClick={async () => {
              await chat.mutateAsync({ message: question.trim() });
            }}
          >
            Ask
          </Button>
        </div>
        {chat.data ? (
          <div className="rounded border border-slate-800 bg-slate-900 p-3 text-sm text-slate-200">
            <p className="text-xs uppercase tracking-wide text-slate-400">Model: {chat.data.model}</p>
            <p className="mt-2 whitespace-pre-wrap">{chat.data.reply}</p>
          </div>
        ) : null}
      </Card>
    </div>
  );
};
