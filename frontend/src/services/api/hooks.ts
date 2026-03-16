import { useMutation, useQuery } from '@tanstack/react-query';

import { queryKeys } from '@/constants/queryKeys';
import { apiBase, requestJson } from '@/services/api/client';
import type {
  AnalyticsLiveResponse,
  BenchmarkResponse,
  CameraListResponse,
  CameraSimpleStatusResponse,
  CameraStreamStatusResponse,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  JobStatusResponse,
  LlmModelsResponse,
  LlmStatusResponse,
  MultiCameraStartRequest,
  MultiCameraStartResponse,
  MultiCameraStopRequest,
  MultiCameraStopResponse,
  ProcessVideoResponse,
  SceneCommentaryResponse,
  SceneEvent,
  SceneObject,
  SceneStateResponse,
  SceneSummaryResponse,
  SceneTimeSummaryResponse,
  ToggleDepthResponse,
  ToggleHeatmapResponse,
  UploadVideoResponse,
  VideoAnalysisResponse,
  VideoChatRequest,
  VideoChatResponse,
} from '@/types/api';

export const useHealth = () =>
  useQuery({
    queryKey: queryKeys.health,
    queryFn: () => requestJson<HealthResponse>('/api/health'),
    refetchInterval: 15_000,
  });

export const useToggleDepth = () =>
  useMutation({
    mutationFn: () => requestJson<ToggleDepthResponse>('/api/toggle-depth', { method: 'POST' }),
  });

export const useBenchmark = () =>
  useQuery({
    queryKey: queryKeys.benchmark,
    queryFn: () => requestJson<BenchmarkResponse>('/api/benchmark'),
    refetchInterval: 20_000,
  });

export const useCameraStatus = () =>
  useQuery({
    queryKey: queryKeys.cameraStatus,
    queryFn: () => requestJson<CameraStreamStatusResponse>('/api/camera-stream/status'),
    refetchInterval: 3_000,
  });

export const useStartCamera = () =>
  useMutation({
    mutationFn: () => requestJson<CameraSimpleStatusResponse>('/api/camera-stream/start', { method: 'POST' }),
  });

export const useStopCamera = () =>
  useMutation({
    mutationFn: () => requestJson<CameraSimpleStatusResponse>('/api/camera-stream/stop', { method: 'POST' }),
  });

export const useToggleHeatmap = () =>
  useMutation({
    mutationFn: () => requestJson<ToggleHeatmapResponse>('/api/camera-stream/toggle-heatmap', { method: 'POST' }),
  });

export const useMultiCameraList = () =>
  useQuery({
    queryKey: queryKeys.cameraList,
    queryFn: () => requestJson<CameraListResponse>('/api/camera/list'),
    refetchInterval: 3_000,
  });

export const useStartMultiCamera = () =>
  useMutation({
    mutationFn: (payload: MultiCameraStartRequest) =>
      requestJson<MultiCameraStartResponse>('/api/camera/start', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
  });

export const useStopMultiCamera = () =>
  useMutation({
    mutationFn: (payload: MultiCameraStopRequest) =>
      requestJson<MultiCameraStopResponse>('/api/camera/stop', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
  });

export const useUploadVideo = () =>
  useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return requestJson<UploadVideoResponse>('/api/upload-video', {
        method: 'POST',
        body: formData,
      });
    },
  });

export const useProcessVideo = () =>
  useMutation({
    mutationFn: (jobId: string) =>
      requestJson<ProcessVideoResponse>(`/api/process-video/${jobId}`, { method: 'POST' }),
  });

export const useJobStatus = (jobId: string | null) =>
  useQuery({
    queryKey: jobId ? queryKeys.jobStatus(jobId) : ['job-status-idle'],
    queryFn: () => requestJson<JobStatusResponse>(`/api/job-status/${jobId}`),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const state = query.state.data as JobStatusResponse | undefined;
      if (!state) {
        return 3_000;
      }
      return state.status === 'processing' || state.status === 'uploaded' ? 3_000 : false;
    },
  });

export const useVideoAnalysis = (jobId: string | null) =>
  useQuery({
    queryKey: jobId ? queryKeys.videoAnalysis(jobId) : ['video-analysis-idle'],
    queryFn: () => requestJson<VideoAnalysisResponse>(`/api/video-analysis/${jobId}`),
    enabled: Boolean(jobId),
  });

export const useVideoChat = (jobId: string | null) =>
  useMutation({
    mutationFn: (payload: VideoChatRequest) => {
      if (!jobId) {
        throw new Error('No job selected.');
      }
      return requestJson<VideoChatResponse>(`/api/video-chat/${jobId}`, {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    },
  });

export const useAnalyticsLive = () =>
  useQuery({
    queryKey: queryKeys.analyticsLive,
    queryFn: () => requestJson<AnalyticsLiveResponse>('/api/analytics/live'),
    refetchInterval: 5_000,
  });

export const useSceneState = () =>
  useQuery({
    queryKey: queryKeys.sceneState,
    queryFn: () => requestJson<SceneStateResponse>('/api/scene/state'),
    refetchInterval: 5_000,
  });

export const useSceneEvents = (limit = 50) =>
  useQuery({
    queryKey: [...queryKeys.sceneEvents, limit],
    queryFn: () => requestJson<SceneEvent[]>(`/api/scene/events?limit=${limit}`),
    refetchInterval: 5_000,
  });

export const useSceneSummary = () =>
  useQuery({
    queryKey: queryKeys.sceneSummary,
    queryFn: () => requestJson<SceneSummaryResponse>('/api/scene/summary'),
    refetchInterval: 5_000,
  });

export const useSceneHistory = (classFilter: string, limit = 100) => {
  const params = new URLSearchParams();
  if (classFilter.trim()) {
    params.set('class_filter', classFilter.trim());
  }
  params.set('limit', String(limit));

  return useQuery({
    queryKey: queryKeys.sceneHistory(classFilter, limit),
    queryFn: () => requestJson<SceneObject[]>(`/api/scene/history?${params.toString()}`),
  });
};

export const useSceneTimeSummary = (minutes = 60) =>
  useQuery({
    queryKey: queryKeys.sceneTimeSummary(minutes),
    queryFn: () => requestJson<SceneTimeSummaryResponse>(`/api/scene/time-summary?minutes=${minutes}`),
  });

export const useSceneCommentary = (limit = 20) =>
  useQuery({
    queryKey: [...queryKeys.sceneCommentary, limit],
    queryFn: () => requestJson<SceneCommentaryResponse>(`/api/scene/commentary?limit=${limit}`),
    refetchInterval: 10_000,
  });

export const useChat = () =>
  useMutation({
    mutationFn: (payload: ChatRequest) =>
      requestJson<ChatResponse>('/api/chat', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
  });

export const useLlmStatus = () =>
  useQuery({
    queryKey: queryKeys.llmStatus,
    queryFn: () => requestJson<LlmStatusResponse>('/api/llm/status'),
    refetchInterval: 15_000,
  });

export const useLlmModels = () =>
  useQuery({
    queryKey: queryKeys.llmModels,
    queryFn: () => requestJson<LlmModelsResponse>('/api/llm/models'),
    refetchInterval: 30_000,
  });

export const useDownloadUrl = (jobId: string | null): string | null =>
  jobId ? `${apiBase}/api/download/${jobId}` : null;

export const useAnalyticsExportUrl = (): string => `${apiBase}/api/analytics/export`;
export const useMjpegUrl = (): string => `${apiBase}/api/camera-stream`;
