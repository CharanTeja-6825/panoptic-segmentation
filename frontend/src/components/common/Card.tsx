import type { HTMLAttributes, PropsWithChildren } from 'react';

import { cn } from '@/utils/classNames';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  className?: string;
  variant?: 'default' | 'elevated' | 'bordered';
}

const variantClasses = {
  default: 'border border-slate-800 bg-slate-900/70 shadow-sm',
  elevated: 'border border-slate-700 bg-slate-900 shadow-lg',
  bordered: 'border-2 border-slate-700 bg-transparent',
};

export const Card = ({ className, variant = 'default', children, ...props }: PropsWithChildren<CardProps>) => (
  <div
    className={cn('rounded-xl p-4', variantClasses[variant], className)}
    {...props}
  >
    {children}
  </div>
);

interface CardHeaderProps {
  className?: string;
}

export const CardHeader = ({ className, children }: PropsWithChildren<CardHeaderProps>) => (
  <div className={cn('flex flex-col space-y-1.5 pb-4', className)}>{children}</div>
);

export const CardTitle = ({ children, className }: PropsWithChildren<{ className?: string }>) => (
  <h3 className={cn('text-sm font-semibold uppercase tracking-wide text-slate-200', className)}>{children}</h3>
);

export const CardDescription = ({ children, className }: PropsWithChildren<{ className?: string }>) => (
  <p className={cn('text-sm text-slate-400', className)}>{children}</p>
);

interface CardContentProps {
  className?: string;
}

export const CardContent = ({ className, children }: PropsWithChildren<CardContentProps>) => (
  <div className={cn('', className)}>{children}</div>
);

interface CardFooterProps {
  className?: string;
}

export const CardFooter = ({ className, children }: PropsWithChildren<CardFooterProps>) => (
  <div className={cn('flex items-center pt-4', className)}>{children}</div>
);
