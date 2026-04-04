import { useState, useRef, useEffect, type PropsWithChildren, type ReactNode } from 'react';

import { cn } from '@/utils/classNames';

type TooltipPosition = 'top' | 'bottom' | 'left' | 'right';

interface TooltipProps {
  content: ReactNode;
  position?: TooltipPosition;
  delay?: number;
  className?: string;
}

const positionClasses: Record<TooltipPosition, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
};

const arrowClasses: Record<TooltipPosition, string> = {
  top: 'top-full left-1/2 -translate-x-1/2 border-t-slate-800 border-x-transparent border-b-transparent',
  bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-slate-800 border-x-transparent border-t-transparent',
  left: 'left-full top-1/2 -translate-y-1/2 border-l-slate-800 border-y-transparent border-r-transparent',
  right: 'right-full top-1/2 -translate-y-1/2 border-r-slate-800 border-y-transparent border-l-transparent',
};

export const Tooltip = ({
  content,
  position = 'top',
  delay = 200,
  className,
  children,
}: PropsWithChildren<TooltipProps>) => {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  const handleMouseEnter = () => {
    timeoutRef.current = window.setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleMouseEnter}
      onBlur={handleMouseLeave}
    >
      {children}
      {isVisible && content && (
        <div
          role="tooltip"
          className={cn(
            'absolute z-50 max-w-xs rounded-md bg-slate-800 px-3 py-1.5 text-xs text-slate-100 shadow-lg',
            'animate-in fade-in-0 zoom-in-95',
            positionClasses[position],
            className,
          )}
        >
          {content}
          <span
            className={cn(
              'absolute h-0 w-0 border-4',
              arrowClasses[position],
            )}
          />
        </div>
      )}
    </div>
  );
};
