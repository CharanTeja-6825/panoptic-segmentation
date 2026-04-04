export interface ApiErrorPayload {
  detail?: string;
  error?: string;
  error_type?: string;
}

export interface HealthResponse {
  status: string;
  device: string;
  gpu: string | null;
  model_size: string;
  depth_estimation: 'enabled' | 'disabled';
  tracking: boolean;
  llm: {
    status: 'connected' | 'disconnected' | 'unavailable';
    vision_model: string;
    fallback_model: string;
    queue_max_size: number;
    total_requests?: number;
    successful_requests?: number;
    timeout_count?: number;
    fallback_count?: number;
    avg_latency_ms?: number;
  };
  commentary: 'enabled' | 'disabled';
  scene_memory: boolean;
}

export interface ToggleDepthResponse {
  depth_enabled: boolean;
}

export interface BenchmarkResponse {
  avg_fps: number;
  gpu_memory_mb: number;
  gpu_memory_total_mb: number;
  gpu_name: string | null;
  cpu_load_1m: number;
  device: string;
  frames_measured: number;
}

export interface CameraStreamStatusResponse {
  running: boolean;
  fps: number;
  camera_index: number;
  heatmap: boolean;
}

export interface CameraSimpleStatusResponse {
  status: string;
}

export interface ToggleHeatmapResponse {
  heatmap: boolean;
}

export interface MultiCameraStartRequest {
  camera_index: number;
  label: string;
}

export interface MultiCameraStartResponse {
  stream_id: string;
  status: 'started';
}

export interface MultiCameraStopRequest {
  stream_id: string;
}

export interface MultiCameraStopResponse {
  stream_id: string;
  status: 'stopped';
}

export interface CameraStreamInfo {
  stream_id: string;
  camera_index: number;
  label: string;
  running: boolean;
  fps: number;
  object_count: number;
}

export interface CameraListResponse {
  streams: CameraStreamInfo[];
}

export interface UploadVideoResponse {
  job_id: string;
  filename: string;
  size_bytes: number;
}

export type JobStatusValue = 'uploaded' | 'processing' | 'done' | 'failed';

export interface ProcessingStats {
  frames_processed: number;
  total_frames: number;
  elapsed_seconds: number;
  average_fps: number;
  output_path: string;
  analysis_path?: string;
  analysis_summary?: {
    total_unique_tracks: number;
    unique_tracks_by_class: Record<string, number>;
    frame_detections_by_class: Record<string, number>;
    events_count: number;
    keyframes_count: number;
  };
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatusValue;
  progress: number;
  total_frames: number;
  fps: number;
  error: string | null;
  download_url?: string;
  stats?: ProcessingStats;
  analysis_url?: string;
}

export interface ProcessVideoResponse {
  job_id: string;
  status: 'processing';
}

export interface AnalysisTimelineCount {
  frame_index: number;
  timestamp_seconds: number;
  total_objects: number;
  counts_by_class: Record<string, number>;
}

export interface AnalysisEvent {
  frame_index: number;
  timestamp_seconds: number;
  event_type: 'entry' | 'exit';
  track_id: number;
  class_name: string;
  confidence?: number;
  duration_seconds?: number;
  duration_frames?: number;
  reason?: string;
}

export interface AnalysisKeyframe {
  frame_index: number;
  timestamp_seconds: number;
  path: string;
  file_name: string;
  url?: string;
}

export interface VideoAnalysisPayload {
  video: {
    input_path: string;
    output_path: string;
    width: number;
    height: number;
    fps: number;
    total_frames: number;
    frames_processed: number;
    duration_seconds: number;
  };
  summary: {
    total_unique_tracks: number;
    unique_tracks_by_class: Record<string, number>;
    frame_detections_by_class: Record<string, number>;
    events_count: number;
    keyframes_count: number;
  };
  timeline: {
    sample_interval_frames: number;
    counts: AnalysisTimelineCount[];
  };
  events: AnalysisEvent[];
  keyframes: AnalysisKeyframe[];
}

export interface VideoAnalysisResponse {
  job_id: string;
  status: 'done';
  analysis: VideoAnalysisPayload;
}

export interface VideoChatRequest {
  message: string;
  model?: string | null;
  temperature?: number;
  max_keyframes?: number;
}

export interface VideoChatResponse {
  job_id: string;
  reply: string;
  model: string;
  used_keyframes: number;
}

export interface AnalyticsEvent {
  timestamp: number;
  event_type: 'entry' | 'exit';
  track_id: number;
  class_name: string;
}

export interface AnalyticsLiveResponse {
  current_counts: Record<string, number>;
  total_objects: number;
  fps: number;
  total_frames: number;
  rolling_avg_counts: Record<string, number>;
  recent_events: AnalyticsEvent[];
}

export interface SceneObject {
  track_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  first_seen: number;
  last_seen: number;
  status: 'active' | 'exited';
}

export interface SceneStateResponse {
  active_objects: SceneObject[];
  active_count: number;
  counts_by_class: Record<string, number>;
  total_unique_objects: number;
  class_totals: Record<string, number>;
}

export interface SceneEvent {
  timestamp: number;
  event_type: 'entry' | 'exit' | 'movement' | 'crowd' | 'idle';
  description: string;
  objects_involved: number[];
  metadata: Record<string, unknown>;
}

export interface SceneSnapshot {
  timestamp: number;
  total_objects: number;
  counts_by_class: Record<string, number>;
  active_track_ids: number[];
  summary_text: string;
}

export interface SceneSummaryResponse {
  timestamp: number;
  active_objects: number;
  counts_by_class: Record<string, number>;
  total_unique_objects_seen: number;
  recent_events: SceneEvent[];
  latest_snapshot: SceneSnapshot | null;
  class_totals: Record<string, number>;
}

export interface SceneTimeSummaryResponse {
  start_time: number;
  end_time: number;
  total_events: number;
  entries: number;
  exits: number;
  entry_classes: Record<string, number>;
}

export interface CommentaryItem {
  timestamp: number;
  text: string;
  type: string;
  priority: 'low' | 'normal' | 'high' | 'critical';
}

export interface SceneCommentaryResponse {
  commentary: CommentaryItem[];
}

export interface ChatRequest {
  message: string;
  model?: string | null;
  temperature?: number;
  include_frame?: boolean;
}

export interface ChatResponse {
  reply: string;
  timestamp: number;
  model: string;
  used_fallback?: boolean;
  from_memory?: boolean;
  queue_size?: number;
  latency_ms?: number;
}

export interface LlmStatusResponse {
  available: boolean;
  base_url?: string;
  model?: string;
  vision_model?: string;
  fallback_model?: string;
  error?: string;
  queue?: {
    size: number;
    max_size: number;
    total_queued: number;
    total_processed: number;
    total_rejected: number;
  };
  metrics?: {
    total_requests: number;
    successful_requests: number;
    timeout_count: number;
    fallback_count: number;
    avg_latency_ms: number;
  };
}

export interface LlmModel {
  name: string;
  size: number;
}

export interface LlmModelsResponse {
  models: LlmModel[];
}

export interface LlmVisionModelsResponse {
  models: LlmModel[];
  default: string;
  fallback: string;
}

export interface LlmMetricsResponse {
  llm: {
    total_requests: number;
    successful_requests: number;
    timeout_count: number;
    fallback_count: number;
    avg_latency_ms: number;
  };
  queue: {
    current_size: number;
    max_size: number;
    total_queued: number;
    total_processed: number;
    total_rejected: number;
  };
}

export interface SceneWsObject {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface SceneWsSceneUpdate {
  type: 'scene_update';
  objects: SceneWsObject[];
}

export interface SceneWsEvent {
  id: string;
  type: string;
  message: string;
  timestamp: string;
  severity: string;
  event_type: string;
  metadata: Record<string, unknown>;
}

export interface SceneWsEventMessage {
  type: 'event';
  event: SceneWsEvent;
}

export type SceneWsMessage = SceneWsSceneUpdate | SceneWsEventMessage;

export interface ChatWsStart {
  type: 'chat_start';
  id: string;
  model: string;
  from_memory?: boolean;
  queue_size?: number;
}

export interface ChatWsToken {
  type: 'chat_token';
  id: string;
  token: string;
}

export interface ChatWsEnd {
  type: 'chat_end';
  id: string;
}

export interface ChatWsError {
  type: 'chat_error';
  error: string;
  error_type?: string;
  queue_size?: number;
}

export type ChatWsMessage = ChatWsStart | ChatWsToken | ChatWsEnd | ChatWsError;
