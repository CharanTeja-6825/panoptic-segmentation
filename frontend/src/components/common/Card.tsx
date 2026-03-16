import type { PropsWithChildren } from 'react';

import { cn } from '@/utils/classNames';

interface CardProps {
  className?: string;
}

export const Card = ({ className, children }: PropsWithChildren<CardProps>) => (
  <section className={cn('rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm', className)}>{children}</section>
);

export const CardTitle = ({ children }: PropsWithChildren) => (
  <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">{children}</h2>
);
