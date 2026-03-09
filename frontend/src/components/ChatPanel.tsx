import { useState, useRef, useEffect } from "react";
import { useStore } from "../store/useStore";
import { chatSocket } from "../services/websocket";
import { useWebSocket } from "../hooks/useWebSocket";

const SUGGESTED_QUERIES = [
  "What objects do you see?",
  "Describe the current scene",
  "Any safety concerns?",
  "Summarize recent activity",
];

export default function ChatPanel() {
  const chatMessages = useStore((s) => s.chatMessages);
  const addChatMessage = useStore((s) => s.addChatMessage);
  const updateChatMessage = useStore((s) => s.updateChatMessage);
  const finishChatMessage = useStore((s) => s.finishChatMessage);
  const chatLoading = useStore((s) => s.chatLoading);
  const setChatLoading = useStore((s) => s.setChatLoading);

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const currentStreamId = useRef<string | null>(null);

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
      setChatLoading(false);
    } else if (msg.type === "chat_token" && currentStreamId.current) {
      updateChatMessage(currentStreamId.current, String(msg.token ?? ""));
    } else if (msg.type === "chat_end" && currentStreamId.current) {
      finishChatMessage(currentStreamId.current);
      currentStreamId.current = null;
    } else if (msg.type === "chat_error") {
      const id = `err-${Date.now()}`;
      addChatMessage({
        id,
        role: "assistant",
        content: `Error: ${String(msg.error ?? "Unknown error")}`,
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
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
    send({ type: "chat", message: text });
    setInput("");
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-700/50 bg-slate-800 shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/50 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <svg
            className="h-4 w-4 text-indigo-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
          <h2 className="text-sm font-semibold text-slate-200">AI Assistant</h2>
        </div>
        <span className="rounded-md bg-slate-700/50 px-2 py-0.5 text-xs text-slate-400">
          {chatMessages.length} messages
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {chatMessages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4">
            <div className="rounded-full bg-indigo-500/10 p-4">
              <svg
                className="h-8 w-8 text-indigo-400"
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
            <p className="text-sm text-slate-400">
              Ask about the scene or detected objects
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTED_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="rounded-full border border-slate-600/50 px-3 py-1.5 text-xs text-slate-300 transition-colors hover:border-indigo-500/50 hover:bg-indigo-500/10 hover:text-indigo-300"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {chatMessages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-700/70 text-slate-200"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  {msg.streaming && (
                    <span className="ml-1 inline-block h-3 w-1.5 animate-pulse rounded-full bg-indigo-400" />
                  )}
                  <div
                    className={`mt-1 text-[10px] ${
                      msg.role === "user"
                        ? "text-indigo-200/60"
                        : "text-slate-500"
                    }`}
                  >
                    {new Date(msg.timestamp).toLocaleTimeString()}
                    {msg.model && ` · ${msg.model}`}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-slate-700/50 p-3">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend(input);
              }
            }}
            placeholder="Ask about the scene..."
            className="flex-1 rounded-lg border border-slate-600/50 bg-slate-700/50 px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 outline-none transition-colors focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/30"
            disabled={chatLoading}
          />
          <button
            onClick={() => handleSend(input)}
            disabled={chatLoading || !input.trim()}
            className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 text-white transition-all hover:bg-indigo-500 disabled:opacity-40 disabled:hover:bg-indigo-600"
          >
            {chatLoading ? (
              <svg
                className="h-4 w-4 animate-spin"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                />
              </svg>
            ) : (
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
