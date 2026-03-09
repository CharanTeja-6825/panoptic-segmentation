import { useEffect, useRef, useCallback } from "react";
import { WebSocketManager } from "../services/websocket";

type MessageHandler = (data: unknown) => void;

export function useWebSocket(
  manager: WebSocketManager,
  event: string,
  handler: MessageHandler
): { connected: boolean; send: (data: Record<string, unknown>) => void } {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    manager.connect();
    const unsubscribe = manager.on(event, (data) => {
      handlerRef.current(data);
    });
    return unsubscribe;
  }, [manager, event]);

  const send = useCallback(
    (data: Record<string, unknown>) => manager.send(data),
    [manager]
  );

  return { connected: manager.connected, send };
}
