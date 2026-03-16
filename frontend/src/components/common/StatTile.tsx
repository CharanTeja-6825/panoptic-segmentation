import { Card } from '@/components/common/Card';

interface StatTileProps {
  label: string;
  value: string | number;
  helper?: string;
}

export const StatTile = ({ label, value, helper }: StatTileProps) => (
  <Card className="space-y-1">
    <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
    <p className="text-2xl font-semibold text-slate-100">{value}</p>
    {helper ? <p className="text-xs text-slate-400">{helper}</p> : null}
  </Card>
);
