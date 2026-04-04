import type { SelectHTMLAttributes } from 'react';

import { cn } from '@/utils/classNames';

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
}

export const Select = ({ className, children, label, error, id, ...props }: SelectProps) => (
  <div className="flex flex-col gap-1.5">
    {label && (
      <label htmlFor={id} className="text-sm font-medium text-slate-300">
        {label}
      </label>
    )}
    <select
      id={id}
      className={cn(
        'flex h-10 w-full items-center justify-between rounded-md border bg-slate-900 px-3 py-2 text-sm',
        'border-slate-700 text-slate-100 placeholder:text-slate-500',
        'focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 focus:ring-offset-slate-900',
        'disabled:cursor-not-allowed disabled:opacity-50',
        error && 'border-rose-500 focus:ring-rose-400',
        className,
      )}
      {...props}
    >
      {children}
    </select>
    {error && <p className="text-xs text-rose-400">{error}</p>}
  </div>
);

// Option component for consistent styling
interface SelectOptionProps extends React.OptionHTMLAttributes<HTMLOptionElement> {
  children: React.ReactNode;
}

export const SelectOption = ({ children, ...props }: SelectOptionProps) => (
  <option className="bg-slate-900 text-slate-100" {...props}>
    {children}
  </option>
);
