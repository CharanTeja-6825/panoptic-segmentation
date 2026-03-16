import type { PropsWithChildren } from 'react';

import { cn } from '@/utils/classNames';

interface BadgeProps {
  tone?: 'default' | 'success' | 'warning' | 'danger';
  className?: string;
}

const toneClasses: Record<NonNullable<BadgeProps['tone']>, string> = {
  default: 'border-slate-600 text-slate-200',
  success: 'border-emerald-600/60 text-emerald-300',
  warning: 'border-amber-500/60 text-amber-300',
  danger: 'border-rose-600/60 text-rose-300',
};

export const Badge = ({ tone = 'default', className, children }: PropsWithChildren<BadgeProps>) => (
  <span className={cn('inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium', toneClasses[tone], className)}>
    {children}
  </span>
);
