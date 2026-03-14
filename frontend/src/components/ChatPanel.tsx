import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store/useStore";
import { chatSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";

const SUGGESTED_QUERIES = [
  "Explain what is happening in this frame.",
  "What risks or anomalies do you detect right now?",
  "Summarize the scene in plain language.",
  "What changed in the last few moments?",
];

export default function ChatPanel() {
  const chatMessages = useStore((s) => s.chatMessages);
  const addChatMessage = useStore((s) => s.addChatMessage);
  const updateChatMessage = useStore((s) => s.updateChatMessage);
  const finishChatMessage = useStore((s) => s.finishChatMessage);
  const chatLoading = useStore((s) => s.chatLoading);
  const setChatLoading = useStore((s) => s.setChatLoading);

  const [input, setInput] = useState("");
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentStreamId = useRef<string | null>(null);
  const shouldAutoScrollRef = useRef(true);

  const { send } = useWebSocket(chatSocket, "message", (data) => {
    const msg = data as Record<string, unknown>;

    if (msg.type === "chat_start") {
      const id = String(msg.id ?? Date.now());
      currentStreamId.current = id;
      addChatMessage({
        id,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        model: String(msg.model ?? ""),
        streaming: true,
      });
      return;
    }

    if (msg.type === "chat_token" && currentStreamId.current) {
      updateChatMessage(currentStreamId.current, String(msg.token ?? ""));
      return;
    }

    if (msg.type === "chat_end" && currentStreamId.current) {
      finishChatMessage(currentStreamId.current);
      currentStreamId.current = null;
      setChatLoading(false);
      return;
    }

    if (msg.type === "chat_error") {
      addChatMessage({
        id: `err-${Date.now()}`,
        role: "assistant",
        content: `Error: ${String(msg.error ?? "Unknown chat error")}`,
        timestamp: new Date().toISOString(),
      });
      setChatLoading(false);
      currentStreamId.current = null;
    }
  });

  useEffect(() => {
    chatSocket.connect();
  }, []);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container || !shouldAutoScrollRef.current) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [chatMessages]);

  const handleSend = (message: string) => {
    const text = message.trim();
    if (!text || chatLoading) return;

    addChatMessage({
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    });

    setChatLoading(true);
    shouldAutoScrollRef.current = true;
    send({ type: "chat", message: text });
    setInput("");
  };

  const renderedMessages = useMemo(() => chatMessages, [chatMessages]);

  return (
    <section className="app-panel">
      <header className="app-panel-header">
        <div>
          <h2 className="app-panel-title">Explainable AI Assistant</h2>
          <p className="mt-0.5 text-xs" style={{ color: "var(--color-text-muted)" }}>
            Grounded in live scene memory &amp; camera frame
          </p>
        </div>
        <span className="status-pill status-pill-muted">
          {renderedMessages.length} messages
        </span>
      </header>

      {/* Messages area */}
      <div
        ref={messagesContainerRef}
        className="min-h-0 flex-1 overflow-y-auto px-4 py-4 [overflow-anchor:none]"
        onScroll={(event) => {
          const element = event.currentTarget;
          const distanceFromBottom =
            element.scrollHeight - element.scrollTop - element.clientHeight;
          shouldAutoScrollRef.current = distanceFromBottom < 80;
        }}
      >
        {renderedMessages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg className="h-7 w-7" style={{ color: "var(--color-accent-hover)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
                Ask about the scene
              </p>
              <p className="mt-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
                Get explanations, risk analysis, or scene summaries.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTED_QUERIES.map((query) => (
                <button
                  key={query}
                  onClick={() => handleSend(query)}
                  className="filter-chip text-left transition-all hover:border-[var(--color-accent)] hover:text-[var(--color-accent-hover)]"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {renderedMessages.map((msg) => (
              <article
                key={msg.id}
                className={`fade-in flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={msg.role === "user" ? "chat-bubble chat-bubble-user" : "chat-bubble chat-bubble-assistant"}>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  <div className="mt-2 flex items-center gap-1.5 text-[11px]" style={{ opacity: 0.7 }}>
                    <span>{new Date(msg.timestamp).toLocaleTimeString()}</span>
                    {msg.model ? <><span>·</span><span>{msg.model}</span></> : null}
                    {msg.streaming ? <><span>·</span><span className="pulse-dot inline-block h-1.5 w-1.5" style={{ background: "var(--color-accent-hover)" }} /></> : null}
                  </div>
                </div>
              </article>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <footer className="p-3" style={{ borderTop: "1px solid var(--color-border)" }}>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend(input);
              }
            }}
            placeholder="Ask the assistant about this scene…"
            className="input-field h-11 flex-1"
            disabled={chatLoading}
          />
          <button
            onClick={() => handleSend(input)}
            disabled={chatLoading || !input.trim()}
            className="btn btn-primary h-11 shrink-0"
          >
            {chatLoading ? (
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            )}
            Send
          </button>
        </div>
      </footer>
    </section>
  );
}
