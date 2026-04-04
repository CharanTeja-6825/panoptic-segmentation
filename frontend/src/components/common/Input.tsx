import type { InputHTMLAttributes } from 'react';

import { cn } from '@/utils/classNames';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Optional label displayed above the input */
  label?: string;
  /** Error message to display below the input */
  error?: string;
}

export const Input = ({ className, label, error, id, ...props }: InputProps) => {
  const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

  return (
    <div className="w-full">
      {label && (
        <label
          htmlFor={inputId}
          className="mb-1.5 block text-sm font-medium text-slate-300"
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={cn(
          'w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100',
          'placeholder:text-slate-500',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 focus-visible:ring-offset-slate-900',
          'disabled:cursor-not-allowed disabled:opacity-50',
          error && 'border-red-500 focus-visible:ring-red-500',
          className,
        )}
        {...props}
      />
      {error && (
        <p className="mt-1.5 text-xs text-red-400">{error}</p>
      )}
    </div>
  );
};
