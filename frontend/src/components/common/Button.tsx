import type { ButtonHTMLAttributes, PropsWithChildren } from 'react';

import { cn } from '@/utils/classNames';

type Variant = 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
type Size = 'default' | 'sm' | 'lg' | 'icon';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

const variantClasses: Record<Variant, string> = {
  default: 'bg-indigo-500 text-white hover:bg-indigo-400 shadow-sm',
  destructive: 'bg-rose-600 text-white hover:bg-rose-500 shadow-sm',
  outline: 'border border-slate-700 bg-transparent text-slate-100 hover:bg-slate-800 hover:text-white',
  secondary: 'bg-slate-700 text-slate-100 hover:bg-slate-600 shadow-sm',
  ghost: 'bg-transparent text-slate-200 hover:bg-slate-800 hover:text-white',
  link: 'bg-transparent text-indigo-400 underline-offset-4 hover:underline',
};

const sizeClasses: Record<Size, string> = {
  default: 'h-10 px-4 py-2',
  sm: 'h-8 px-3 py-1 text-xs',
  lg: 'h-12 px-6 py-3',
  icon: 'h-10 w-10 p-0',
};

export const Button = ({
  variant = 'default',
  size = 'default',
  loading = false,
  className,
  children,
  disabled,
  ...props
}: PropsWithChildren<ButtonProps>) => (
  <button
    className={cn(
      'inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium transition-colors',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900',
      'disabled:pointer-events-none disabled:opacity-50',
      variantClasses[variant],
      sizeClasses[size],
      className,
    )}
    disabled={disabled || loading}
    {...props}
  >
    {loading && (
      <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    )}
    {children}
  </button>
);

// Backward compatibility aliases
export const ButtonPrimary = (props: PropsWithChildren<Omit<ButtonProps, 'variant'>>) => (
  <Button variant="default" {...props} />
);

export const ButtonSecondary = (props: PropsWithChildren<Omit<ButtonProps, 'variant'>>) => (
  <Button variant="secondary" {...props} />
);

export const ButtonDanger = (props: PropsWithChildren<Omit<ButtonProps, 'variant'>>) => (
  <Button variant="destructive" {...props} />
);

export const ButtonGhost = (props: PropsWithChildren<Omit<ButtonProps, 'variant'>>) => (
  <Button variant="ghost" {...props} />
);
