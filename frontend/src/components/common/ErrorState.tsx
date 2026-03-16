import type { PropsWithChildren } from 'react';

interface ErrorStateProps {
  title?: string;
}

export const ErrorState = ({ title = 'Request failed', children }: PropsWithChildren<ErrorStateProps>) => (
  <div className="rounded-lg border border-rose-500/70 bg-rose-950/20 p-3 text-sm text-rose-200">
    <p className="font-semibold">{title}</p>
    {children ? <p className="mt-1">{children}</p> : null}
  </div>
);
