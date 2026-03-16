import { useMemo, useState } from 'react';

import { Button } from '@/components/common/Button';
import { Card, CardTitle } from '@/components/common/Card';
import { ErrorState } from '@/components/common/ErrorState';
import { Input } from '@/components/common/Input';
import { Select } from '@/components/common/Select';
import { PageHeader } from '@/components/layout/PageHeader';
import { useChatStream } from '@/hooks/useChatStream';
import { useChat, useLlmModels, useLlmStatus } from '@/services/api/hooks';

interface LocalMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export const ChatPage = () => {
  const [message, setMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [history, setHistory] = useState<LocalMessage[]>([]);

  const chat = useChat();
  const llmStatus = useLlmStatus();
  const llmModels = useLlmModels();
  const stream = useChatStream();

  const sortedModels = useMemo(() => llmModels.data?.models ?? [], [llmModels.data?.models]);

  return (
    <div className="space-y-4">
      <PageHeader title="LLM chat" subtitle="Ask scene-aware questions with standard or streaming responses" />

      {(chat.error || llmStatus.error || llmModels.error) ? (
        <ErrorState>Chat service returned an error.</ErrorState>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="space-y-3">
          <CardTitle>Non-streaming chat (`POST /api/chat`)</CardTitle>
          <div className="grid gap-2 sm:grid-cols-[1fr,220px]">
            <Input value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Describe the current scene..." />
            <Select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              <option value="">Default model</option>
              {sortedModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}
                </option>
              ))}
            </Select>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={async () => {
                const trimmed = message.trim();
                if (!trimmed) {
                  return;
                }
                const response = await chat.mutateAsync({ message: trimmed, model: selectedModel || null });
                setHistory((items) => [
                  ...items,
                  { id: `${Date.now()}-u`, role: 'user', content: trimmed },
                  { id: `${Date.now()}-a`, role: 'assistant', content: response.reply },
                ]);
                setMessage('');
              }}
              disabled={chat.isPending}
            >
              Send
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                const trimmed = message.trim();
                if (!trimmed) {
                  return;
                }
                stream.send(trimmed, selectedModel || undefined);
              }}
              disabled={stream.isStreaming}
            >
              Stream
            </Button>
          </div>

          <div className="space-y-2">
            {history.map((item) => (
              <div key={item.id} className={`rounded p-3 text-sm ${item.role === 'user' ? 'bg-indigo-500/20 text-indigo-100' : 'bg-slate-800 text-slate-200'}`}>
                <p className="text-xs uppercase tracking-wide opacity-75">{item.role}</p>
                <p className="mt-1 whitespace-pre-wrap">{item.content}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card className="space-y-3">
          <CardTitle>Streaming output (`WS /api/chat/stream`)</CardTitle>
          <dl className="space-y-1 text-sm text-slate-300">
            <div className="flex justify-between"><dt>LLM available</dt><dd>{String(llmStatus.data?.available ?? false)}</dd></div>
            <div className="flex justify-between"><dt>Base URL</dt><dd>{llmStatus.data?.base_url ?? 'n/a'}</dd></div>
            <div className="flex justify-between"><dt>Default model</dt><dd>{llmStatus.data?.model ?? 'n/a'}</dd></div>
          </dl>

          {stream.error ? <ErrorState>{stream.error}</ErrorState> : null}

          <div className="rounded border border-slate-800 bg-slate-900 p-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Stream model: {stream.model ?? 'n/a'}</p>
            <p className="mt-2 min-h-40 whitespace-pre-wrap text-sm text-slate-100">{stream.output || 'No stream output yet.'}</p>
          </div>
        </Card>
      </div>
    </div>
  );
};
