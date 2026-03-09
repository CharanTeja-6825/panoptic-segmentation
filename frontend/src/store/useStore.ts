import { create } from "zustand";

export interface SceneObject {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface SceneEvent {
  id: string;
  type: "detection" | "alert" | "info" | "warning" | "system";
  message: string;
  timestamp: string;
  severity?: "low" | "medium" | "high";
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  model?: string;
  streaming?: boolean;
}

export interface SystemHealth {
  api: boolean;
  camera: boolean;
  llm: boolean;
  websocket: boolean;
}

export interface AnalyticsData {
  totalDetections: number;
  objectCounts: Record<string, number>;
  fps: number;
  latency: number;
  uptime: number;
}

interface AppState {
  // Scene
  sceneObjects: SceneObject[];
  sceneEvents: SceneEvent[];
  sceneSummary: string;

  // Chat
  chatMessages: ChatMessage[];
  chatLoading: boolean;

  // System
  systemHealth: SystemHealth;
  cameraActive: boolean;
  darkMode: boolean;
  detectionSensitivity: number;
  recording: boolean;

  // Analytics
  analytics: AnalyticsData;
  availableModels: string[];
  selectedModel: string;

  // Actions
  setSceneObjects: (objects: SceneObject[]) => void;
  addSceneEvent: (event: SceneEvent) => void;
  setSceneEvents: (events: SceneEvent[]) => void;
  setSceneSummary: (summary: string) => void;

  addChatMessage: (message: ChatMessage) => void;
  updateChatMessage: (id: string, content: string) => void;
  finishChatMessage: (id: string) => void;
  setChatLoading: (loading: boolean) => void;

  setSystemHealth: (health: Partial<SystemHealth>) => void;
  setCameraActive: (active: boolean) => void;
  toggleDarkMode: () => void;
  setDetectionSensitivity: (value: number) => void;
  setRecording: (recording: boolean) => void;

  setAnalytics: (data: Partial<AnalyticsData>) => void;
  setAvailableModels: (models: string[]) => void;
  setSelectedModel: (model: string) => void;
}

export const useStore = create<AppState>((set) => ({
  sceneObjects: [],
  sceneEvents: [],
  sceneSummary: "",

  chatMessages: [],
  chatLoading: false,

  systemHealth: {
    api: false,
    camera: false,
    llm: false,
    websocket: false,
  },
  cameraActive: false,
  darkMode: true,
  detectionSensitivity: 50,
  recording: false,

  analytics: {
    totalDetections: 0,
    objectCounts: {},
    fps: 0,
    latency: 0,
    uptime: 0,
  },
  availableModels: [],
  selectedModel: "",

  setSceneObjects: (objects) => set({ sceneObjects: objects }),
  addSceneEvent: (event) =>
    set((state) => ({
      sceneEvents: [event, ...state.sceneEvents].slice(0, 200),
    })),
  setSceneEvents: (events) => set({ sceneEvents: events }),
  setSceneSummary: (summary) => set({ sceneSummary: summary }),

  addChatMessage: (message) =>
    set((state) => ({
      chatMessages: [...state.chatMessages, message],
    })),
  updateChatMessage: (id, content) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((m) =>
        m.id === id ? { ...m, content: m.content + content } : m
      ),
    })),
  finishChatMessage: (id) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((m) =>
        m.id === id ? { ...m, streaming: false } : m
      ),
    })),
  setChatLoading: (loading) => set({ chatLoading: loading }),

  setSystemHealth: (health) =>
    set((state) => ({
      systemHealth: { ...state.systemHealth, ...health },
    })),
  setCameraActive: (active) => set({ cameraActive: active }),
  toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
  setDetectionSensitivity: (value) => set({ detectionSensitivity: value }),
  setRecording: (recording) => set({ recording }),

  setAnalytics: (data) =>
    set((state) => ({
      analytics: { ...state.analytics, ...data },
    })),
  setAvailableModels: (models) => set({ availableModels: models }),
  setSelectedModel: (model) => set({ selectedModel: model }),
}));
