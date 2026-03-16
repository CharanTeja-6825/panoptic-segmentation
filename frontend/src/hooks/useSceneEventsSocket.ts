import { useEffect } from 'react';

import { toWsUrl } from '@/services/api/client';
import { useWsStore } from '@/store/wsStore';
import type { SceneWsMessage } from '@/types/api';

const isSceneWsMessage = (value: unknown): value is SceneWsMessage => {
  if (typeof value !== 'object' || value === null || !('type' in value)) {
    return false;
  }
  const payload = value as { type: unknown };
  return payload.type === 'scene_update' || payload.type === 'event';
};

export const useSceneEventsSocket = (): void => {
  const { setConnected, pushEvent, pushObjects } = useWsStore();

  useEffect(() => {
    const ws = new WebSocket(toWsUrl('/api/ws/events'));

    ws.onopen = () => {
      setConnected(true);
      ws.send('subscribe');
    };

    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        return;
      }

      if (!isSceneWsMessage(parsed)) {
        return;
      }

      if (parsed.type === 'scene_update') {
        pushObjects(parsed.objects);
      }

      if (parsed.type === 'event') {
        pushEvent(parsed.event);
      }
    };

    const keepAlive = window.setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 15_000);

    return () => {
      clearInterval(keepAlive);
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [pushEvent, pushObjects, setConnected]);
};
