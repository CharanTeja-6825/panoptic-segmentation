import { create } from 'zustand';

import type { SceneWsEvent, SceneWsObject } from '@/types/api';

interface WsState {
  connected: boolean;
  latestObjects: SceneWsObject[];
  latestEvents: SceneWsEvent[];
  setConnected: (value: boolean) => void;
  pushObjects: (items: SceneWsObject[]) => void;
  pushEvent: (event: SceneWsEvent) => void;
  clear: () => void;
}

export const useWsStore = create<WsState>((set) => ({
  connected: false,
  latestObjects: [],
  latestEvents: [],
  setConnected: (value) => set({ connected: value }),
  pushObjects: (items) => set({ latestObjects: items }),
  pushEvent: (event) =>
    set((state) => ({
      latestEvents: [event, ...state.latestEvents].slice(0, 200),
    })),
  clear: () => set({ latestObjects: [], latestEvents: [] }),
}));
