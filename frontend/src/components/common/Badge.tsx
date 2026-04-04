import type { HTMLAttributes, PropsWithChildren } from 'react';

import { cn } from '@/utils/classNames';

type BadgeVariant = 'default' | 'secondary' | 'success' | 'warning' | 'destructive' | 'outline' | 'info';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: 'default' | 'sm' | 'lg';
  dot?: boolean;
}

const variantClasses: Record<BadgeVariant, string> = {
  default: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
  secondary: 'bg-slate-700/50 text-slate-300 border-slate-600',
  success: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  warning: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  destructive: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
  outline: 'bg-transparent text-slate-300 border-slate-600',
  info: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
};

const sizeClasses = {
  default: 'px-2.5 py-0.5 text-xs',
  sm: 'px-2 py-0.5 text-[10px]',
  lg: 'px-3 py-1 text-sm',
};

const dotColors: Record<BadgeVariant, string> = {
  default: 'bg-indigo-400',
  secondary: 'bg-slate-400',
  success: 'bg-emerald-400',
  warning: 'bg-amber-400',
  destructive: 'bg-rose-400',
  outline: 'bg-slate-400',
  info: 'bg-blue-400',
};

export const Badge = ({
  variant = 'default',
  size = 'default',
  dot = false,
  className,
  children,
  ...props
}: PropsWithChildren<BadgeProps>) => (
  <span
    className={cn(
      'inline-flex items-center gap-1.5 rounded-full border font-medium',
      variantClasses[variant],
      sizeClasses[size],
      className,
    )}
    {...props}
  >
    {dot && <span className={cn('h-1.5 w-1.5 rounded-full', dotColors[variant])} />}
    {children}
  </span>
);

// Backward compatibility alias
export type { BadgeVariant as BadgeTone };
export const toneClasses = variantClasses;
