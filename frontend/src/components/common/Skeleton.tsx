import { cn } from '@/utils/classNames';

export const Skeleton = ({ className }: { className?: string }) => (
  <div className={cn('animate-pulse rounded-md bg-slate-800/80', className)} aria-hidden="true" />
);
