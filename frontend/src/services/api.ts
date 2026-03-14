const BASE_URL = "";

interface ChatRequest {
  message: string;
  model?: string;
  temperature?: number;
}

interface ChatResponse {
  reply: string;
  timestamp: string;
  model: string;
}

interface HealthResponse {
  status: string;
  [key: string]: unknown;
}

interface SceneState {
  objects: Array<{
    id: string;
    label: string;
    confidence: number;
    bbox: [number, number, number, number];
  }>;
  [key: string]: unknown;
}

interface LLMStatus {
  available: boolean;
  model?: string;
  [key: string]: unknown;
}

interface JobStatus {
  job_id: string;
  status: string;
  progress?: number;
  [key: string]: unknown;
}

interface BenchmarkData {
  fps: number;
  latency: number;
  [key: string]: unknown;
}

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

// Health
export function getHealth(): Promise<HealthResponse> {
  return request("/api/health");
}

// Camera
export function startCamera(): Promise<{ status: string }> {
  return request("/api/camera-stream/start", { method: "POST" });
}

export function stopCamera(): Promise<{ status: string }> {
  return request("/api/camera-stream/stop", { method: "POST" });
}

export function getCameraStreamUrl(): string {
  return `${BASE_URL}/api/camera-stream`;
}

// Analytics
export function getLiveAnalytics(): Promise<Record<string, unknown>> {
  return request("/api/analytics/live");
}

export function exportAnalytics(): string {
  return `${BASE_URL}/api/analytics/export`;
}

// Chat
export function sendChatMessage(data: ChatRequest): Promise<ChatResponse> {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// Scene
export function getSceneState(): Promise<SceneState> {
  return request("/api/scene/state");
}

export function getSceneEvents(): Promise<Array<Record<string, unknown>>> {
  return request("/api/scene/events");
}

export function getSceneSummary(): Promise<{ summary: string }> {
  return request("/api/scene/summary");
}

export function getSceneCommentary(): Promise<{ commentary: unknown[] }> {
  return request("/api/scene/commentary");
}

// LLM
export function getLLMStatus(): Promise<LLMStatus> {
  return request("/api/llm/status");
}

export function getLLMModels(): Promise<{ models: Array<{ name: string; size: number }> }> {
  return request("/api/llm/models");
}

// Video
export function uploadVideo(file: File): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  return fetch(`${BASE_URL}/api/upload-video`, {
    method: "POST",
    body: formData,
  }).then((r) => {
    if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
    return r.json() as Promise<{ job_id: string }>;
  });
}

export function processVideo(jobId: string): Promise<{ status: string }> {
  return request(`/api/process-video/${jobId}`, { method: "POST" });
}

export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request(`/api/job-status/${jobId}`);
}

// Benchmark
export function getBenchmark(): Promise<BenchmarkData> {
  return request("/api/benchmark");
}
