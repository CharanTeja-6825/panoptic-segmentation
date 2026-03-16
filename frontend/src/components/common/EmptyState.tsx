interface EmptyStateProps {
  title: string;
  description?: string;
}

export const EmptyState = ({ title, description }: EmptyStateProps) => (
  <div className="rounded-lg border border-dashed border-slate-700 p-5 text-center text-sm text-slate-400">
    <p className="font-medium text-slate-200">{title}</p>
    {description ? <p className="mt-1">{description}</p> : null}
  </div>
);
