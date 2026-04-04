import type { HTMLAttributes, PropsWithChildren, ReactNode } from 'react';

import { cn } from '@/utils/classNames';

type AlertVariant = 'default' | 'info' | 'success' | 'warning' | 'destructive';

interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
  icon?: ReactNode;
}

const variantClasses: Record<AlertVariant, string> = {
  default: 'border-slate-700 bg-slate-900 text-slate-100',
  info: 'border-blue-500/50 bg-blue-500/10 text-blue-200',
  success: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-200',
  warning: 'border-amber-500/50 bg-amber-500/10 text-amber-200',
  destructive: 'border-rose-500/50 bg-rose-500/10 text-rose-200',
};

const iconColors: Record<AlertVariant, string> = {
  default: 'text-slate-400',
  info: 'text-blue-400',
  success: 'text-emerald-400',
  warning: 'text-amber-400',
  destructive: 'text-rose-400',
};

export const Alert = ({
  variant = 'default',
  icon,
  className,
  children,
  ...props
}: PropsWithChildren<AlertProps>) => (
  <div
    role="alert"
    className={cn(
      'relative w-full rounded-lg border p-4',
      variantClasses[variant],
      className,
    )}
    {...props}
  >
    <div className="flex gap-3">
      {icon && <span className={cn('flex-shrink-0', iconColors[variant])}>{icon}</span>}
      <div className="flex-1">{children}</div>
    </div>
  </div>
);

export const AlertTitle = ({ children, className }: PropsWithChildren<{ className?: string }>) => (
  <h5 className={cn('mb-1 font-medium leading-none tracking-tight', className)}>{children}</h5>
);

export const AlertDescription = ({ children, className }: PropsWithChildren<{ className?: string }>) => (
  <div className={cn('text-sm opacity-90', className)}>{children}</div>
);

// Pre-built alert icons
export const AlertIcons = {
  Info: () => (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  Success: () => (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  Warning: () => (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  Error: () => (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
};
