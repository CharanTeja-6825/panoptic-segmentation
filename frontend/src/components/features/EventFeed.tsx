interface EventFeedItem {
  id: string;
  message: string;
  timestamp: string;
  severity?: string;
}

interface EventFeedProps {
  items: EventFeedItem[];
}

export const EventFeed = ({ items }: EventFeedProps) => {
  if (!items.length) {
    return <p className="text-sm text-slate-400">No events yet.</p>;
  }

  return (
    <ul className="max-h-96 space-y-2 overflow-auto pr-1 scrollbar-thin">
      {items.map((item) => (
        <li key={item.id} className="rounded-lg border border-slate-800 bg-slate-900 p-3 text-sm">
          <div className="flex items-center justify-between gap-2">
            <p className="font-medium text-slate-200">{item.message}</p>
            <span className="text-xs text-slate-400">{item.timestamp}</span>
          </div>
          {item.severity ? <p className="mt-1 text-xs uppercase tracking-wide text-slate-500">severity: {item.severity}</p> : null}
        </li>
      ))}
    </ul>
  );
};
