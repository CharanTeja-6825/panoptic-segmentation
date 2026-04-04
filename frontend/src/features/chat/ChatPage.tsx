import { useMemo, useState } from 'react';

import { Alert, AlertDescription, AlertIcons } from '@/components/common/Alert';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';
import { ErrorState } from '@/components/common/ErrorState';
import { Input } from '@/components/common/Input';
import { Select } from '@/components/common/Select';
import { Tooltip } from '@/components/common/Tooltip';
import { PageHeader } from '@/components/layout/PageHeader';
import { useChatStream } from '@/hooks/useChatStream';
import { useChat, useLlmVisionModels, useLlmStatus, useLlmMetrics } from '@/services/api/hooks';

interface LocalMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  fromMemory?: boolean;
  model?: string;
}

export const ChatPage = () => {
  const [message, setMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [history, setHistory] = useState<LocalMessage[]>([]);

  const chat = useChat();
  const llmStatus = useLlmStatus();
  const llmVisionModels = useLlmVisionModels();
  const llmMetrics = useLlmMetrics();
  const stream = useChatStream();

  const visionModels = useMemo(() => llmVisionModels.data?.models ?? [], [llmVisionModels.data?.models]);
  const defaultVisionModel = llmVisionModels.data?.default ?? 'llava-phi3';

  const queueStatus = llmMetrics.data?.queue;
  const isQueueBusy = queueStatus && queueStatus.current_size >= queueStatus.max_size;

  const handleSend = async () => {
    const trimmed = message.trim();
    if (!trimmed) return;

    try {
      const response = await chat.mutateAsync({
        message: trimmed,
        model: selectedModel || null,
      });

      setHistory((items) => [
        ...items,
        { id: `${Date.now()}-u`, role: 'user', content: trimmed },
        {
          id: `${Date.now()}-a`,
          role: 'assistant',
          content: response.reply,
          fromMemory: response.from_memory,
          model: response.model,
        },
      ]);
      setMessage('');
    } catch (error) {
      // Error handled by mutation
    }
  };

  const handleStream = () => {
    const trimmed = message.trim();
    if (!trimmed) return;
    stream.send(trimmed, selectedModel || undefined);
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title="Scene AI Chat"
        subtitle="Ask questions about what the camera is watching"
      />

      {/* Queue status alert */}
      {isQueueBusy && (
        <Alert variant="warning" icon={<AlertIcons.Warning />}>
          <AlertDescription>
            LLM queue is full ({queueStatus?.current_size}/{queueStatus?.max_size}). 
            Requests may be delayed or rejected.
          </AlertDescription>
        </Alert>
      )}

      {(chat.error || llmStatus.error || llmVisionModels.error) && (
        <ErrorState>Chat service returned an error. Check LLM connection.</ErrorState>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        {/* Chat panel */}
        <Card>
          <CardHeader>
            <CardTitle>Vision Chat</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {/* Model selector */}
            <div className="grid gap-2 sm:grid-cols-[1fr,200px]">
              <Input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="What do you see in the camera?"
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              />
              <Tooltip content="Select vision model for image analysis">
                <Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  label=""
                >
                  <option value="">Default ({defaultVisionModel})</option>
                  {visionModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </Select>
              </Tooltip>
            </div>

            {/* Action buttons */}
            <div className="flex gap-2">
              <Button
                onClick={handleSend}
                loading={chat.isPending}
                disabled={!message.trim() || isQueueBusy}
              >
                Send
              </Button>
              <Button
                variant="secondary"
                onClick={handleStream}
                disabled={stream.isStreaming || !message.trim()}
              >
                Stream
              </Button>
            </div>

            {/* Chat history */}
            <div className="max-h-96 space-y-2 overflow-y-auto scrollbar-thin">
              {history.map((item) => (
                <div
                  key={item.id}
                  className={`rounded-lg p-3 text-sm ${
                    item.role === 'user'
                      ? 'bg-indigo-500/20 text-indigo-100'
                      : 'bg-slate-800 text-slate-200'
                  }`}
                >
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-xs font-medium uppercase tracking-wide opacity-75">
                      {item.role}
                    </span>
                    {item.fromMemory && (
                      <Badge variant="info" size="sm">From Memory</Badge>
                    )}
                    {item.model && item.role === 'assistant' && (
                      <Badge variant="secondary" size="sm">{item.model}</Badge>
                    )}
                  </div>
                  <p className="whitespace-pre-wrap">{item.content}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Status panel */}
        <Card>
          <CardHeader>
            <CardTitle>LLM Status & Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Connection status */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-slate-300">Connection</h4>
              <dl className="grid grid-cols-2 gap-2 text-sm">
                <dt className="text-slate-400">Status</dt>
                <dd>
                  <Badge variant={llmStatus.data?.available ? 'success' : 'destructive'} dot>
                    {llmStatus.data?.available ? 'Connected' : 'Disconnected'}
                  </Badge>
                </dd>
                <dt className="text-slate-400">Vision Model</dt>
                <dd className="text-slate-200">{llmStatus.data?.vision_model ?? defaultVisionModel}</dd>
                <dt className="text-slate-400">Fallback</dt>
                <dd className="text-slate-200">{llmStatus.data?.fallback_model ?? 'llava:7b'}</dd>
              </dl>
            </div>

            {/* Queue metrics */}
            {queueStatus && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-slate-300">Queue</h4>
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <dt className="text-slate-400">Current Size</dt>
                  <dd>
                    <Badge variant={queueStatus.current_size > 0 ? 'warning' : 'secondary'}>
                      {queueStatus.current_size} / {queueStatus.max_size}
                    </Badge>
                  </dd>
                  <dt className="text-slate-400">Processed</dt>
                  <dd className="text-slate-200">{queueStatus.total_processed}</dd>
                  <dt className="text-slate-400">Rejected</dt>
                  <dd className="text-slate-200">{queueStatus.total_rejected}</dd>
                </dl>
              </div>
            )}

            {/* Performance metrics */}
            {llmMetrics.data?.llm && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-slate-300">Performance</h4>
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <dt className="text-slate-400">Avg Latency</dt>
                  <dd className="text-slate-200">{llmMetrics.data.llm.avg_latency_ms.toFixed(0)}ms</dd>
                  <dt className="text-slate-400">Success Rate</dt>
                  <dd className="text-slate-200">
                    {llmMetrics.data.llm.total_requests > 0
                      ? ((llmMetrics.data.llm.successful_requests / llmMetrics.data.llm.total_requests) * 100).toFixed(1)
                      : 0}%
                  </dd>
                  <dt className="text-slate-400">Timeouts</dt>
                  <dd className="text-slate-200">{llmMetrics.data.llm.timeout_count}</dd>
                  <dt className="text-slate-400">Fallbacks</dt>
                  <dd className="text-slate-200">{llmMetrics.data.llm.fallback_count}</dd>
                </dl>
              </div>
            )}

            {/* Streaming output */}
            {stream.output && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-slate-300">Stream Output</h4>
                <div className="rounded-lg border border-slate-700 bg-slate-900 p-3">
                  <p className="text-xs text-slate-500">Model: {stream.model ?? 'n/a'}</p>
                  <p className="mt-2 min-h-20 whitespace-pre-wrap text-sm text-slate-100">
                    {stream.output}
                  </p>
                </div>
              </div>
            )}

            {stream.error && (
              <Alert variant="destructive" icon={<AlertIcons.Error />}>
                <AlertDescription>{stream.error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
