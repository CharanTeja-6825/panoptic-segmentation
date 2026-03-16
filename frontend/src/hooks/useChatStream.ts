import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { toWsUrl } from '@/services/api/client';
import type { ChatWsMessage } from '@/types/api';

interface StreamState {
  streamId: string | null;
  model: string | null;
  output: string;
  isStreaming: boolean;
  error: string | null;
}

const isChatWsMessage = (value: unknown): value is ChatWsMessage => {
  if (typeof value !== 'object' || value === null || !('type' in value)) {
    return false;
  }
  const t = (value as { type: unknown }).type;
  return t === 'chat_start' || t === 'chat_token' || t === 'chat_end' || t === 'chat_error';
};

export const useChatStream = () => {
  const socketRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<StreamState>({
    streamId: null,
    model: null,
    output: '',
    isStreaming: false,
    error: null,
  });

  useEffect(() => {
    const ws = new WebSocket(toWsUrl('/api/chat/stream'));
    socketRef.current = ws;

    ws.onmessage = (event) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        return;
      }

      if (!isChatWsMessage(parsed)) {
        return;
      }

      if (parsed.type === 'chat_start') {
        setState({ streamId: parsed.id, model: parsed.model, output: '', isStreaming: true, error: null });
      } else if (parsed.type === 'chat_token') {
        setState((current) => ({ ...current, output: current.output + parsed.token, isStreaming: true }));
      } else if (parsed.type === 'chat_end') {
        setState((current) => ({ ...current, streamId: parsed.id, isStreaming: false }));
      } else if (parsed.type === 'chat_error') {
        setState((current) => ({ ...current, isStreaming: false, error: parsed.error }));
      }
    };

    ws.onerror = () => {
      setState((current) => ({ ...current, isStreaming: false, error: 'Streaming connection failed.' }));
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  const send = useCallback((message: string, model?: string, temperature = 0.7) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setState((current) => ({ ...current, error: 'Streaming socket not connected.' }));
      return;
    }

    socket.send(
      JSON.stringify({
        message,
        model,
        temperature,
      }),
    );
  }, []);

  const api = useMemo(
    () => ({
      ...state,
      send,
    }),
    [send, state],
  );

  return api;
};
