const RECONNECT_BACKOFF_MULTIPLIER = 1.5;

type MessageHandler = (data: unknown) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 2000;
  private maxReconnectDelay = 30000;
  private currentDelay: number;
  private shouldReconnect = true;

  constructor(url: string) {
    this.url = url;
    this.currentDelay = this.reconnectDelay;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const fullUrl = `${protocol}//${window.location.host}${this.url}`;

    this.ws = new WebSocket(fullUrl);

    this.ws.onopen = () => {
      this.currentDelay = this.reconnectDelay;
      this.emit("connection", { status: "connected" });
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as Record<string, unknown>;
        const type = (data.type as string) ?? "message";
        this.emit(type, data);
        this.emit("message", data);
      } catch {
        this.emit("message", { raw: event.data });
      }
    };

    this.ws.onclose = () => {
      this.emit("connection", { status: "disconnected" });
      if (this.shouldReconnect) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.emit("error", { message: "WebSocket error" });
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
      this.currentDelay = Math.min(
        this.currentDelay * RECONNECT_BACKOFF_MULTIPLIER,
        this.maxReconnectDelay
      );
    }, this.currentDelay);
  }

  send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  on(event: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    return () => {
      this.handlers.get(event)?.delete(handler);
    };
  }

  private emit(event: string, data: unknown): void {
    this.handlers.get(event)?.forEach((handler) => handler(data));
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instances
export const eventSocket = new WebSocketManager("/api/ws/events");
export const chatSocket = new WebSocketManager("/api/chat/stream");
