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
          <p className="mt-0.5 text-xs text-slate-400">
            Asks are grounded in live scene memory and latest camera frame
          </p>
        </div>
        <span className="status-pill status-pill-muted">
          {renderedMessages.length} messages
        </span>
      </header>

      <div
        ref={messagesContainerRef}
        className="min-h-0 flex-1 overflow-y-auto px-4 py-3 [overflow-anchor:none]"
        onScroll={(event) => {
          const element = event.currentTarget;
          const distanceFromBottom =
            element.scrollHeight - element.scrollTop - element.clientHeight;
          shouldAutoScrollRef.current = distanceFromBottom < 80;
        }}
      >
        {renderedMessages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="rounded-full border border-slate-700 bg-slate-900 p-4">
              <svg
                className="h-8 w-8 text-indigo-300"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
                />
              </svg>
            </div>
            <p className="text-sm text-slate-300">
              Ask for explanation, risk analysis, or scene summaries.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTED_QUERIES.map((query) => (
                <button
                  key={query}
                  onClick={() => handleSend(query)}
                  className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-200 transition-colors hover:border-indigo-500/50 hover:bg-indigo-500/10 hover:text-indigo-200"
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
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[86%] rounded-2xl px-4 py-3 text-sm shadow ${
                    msg.role === "user"
                      ? "bg-indigo-600 text-white shadow-indigo-900/40"
                      : "border border-slate-700 bg-slate-800 text-slate-100 shadow-slate-950/30"
                  }`}
                >
                  <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  <div
                    className={`mt-2 text-[11px] ${
                      msg.role === "user" ? "text-indigo-200" : "text-slate-400"
                    }`}
                  >
                    {new Date(msg.timestamp).toLocaleTimeString()}
                    {msg.model ? ` · ${msg.model}` : ""}
                    {msg.streaming ? " · typing…" : ""}
                  </div>
                </div>
              </article>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <footer className="border-t border-slate-700/60 p-3">
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
            placeholder="Ask the assistant about this scene..."
            className="h-11 flex-1 rounded-xl border border-slate-700 bg-slate-900 px-4 text-sm text-slate-100 outline-none transition-colors focus:border-indigo-500/70 focus:ring-2 focus:ring-indigo-500/20"
            disabled={chatLoading}
          />
          <button
            onClick={() => handleSend(input)}
            disabled={chatLoading || !input.trim()}
            className="h-11 rounded-xl bg-indigo-600 px-4 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {chatLoading ? "Sending..." : "Send"}
          </button>
        </div>
      </footer>
    </section>
  );
}
