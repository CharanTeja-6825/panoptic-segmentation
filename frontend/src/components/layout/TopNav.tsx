import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { useUiStore } from '@/store/uiStore';
import { useWsStore } from '@/store/wsStore';

export const TopNav = () => {
  const { connected } = useWsStore();
  const { theme, toggleTheme } = useUiStore();

  return (
    <header className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-4 py-3 backdrop-blur-sm">
      <div>
        <p className="text-sm font-medium text-slate-100">Live Monitoring Dashboard</p>
        <p className="text-xs text-slate-400">FastAPI + YOLOv8 + LLM</p>
      </div>
      <div className="flex items-center gap-2">
        <Badge tone={connected ? 'success' : 'warning'}>{connected ? 'WS connected' : 'WS reconnecting'}</Badge>
        <Button variant="ghost" onClick={toggleTheme} aria-label="Toggle theme">
          Theme: {theme}
        </Button>
      </div>
    </header>
  );
};
