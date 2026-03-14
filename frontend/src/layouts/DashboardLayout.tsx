import { ReactNode } from "react";
import TopNav from "../components/TopNav";

interface DashboardLayoutProps {
  children: ReactNode;
  gridClassName?: string;
}

export default function DashboardLayout({ children, gridClassName }: DashboardLayoutProps) {
  const baseGrid =
    "grid min-h-full grid-cols-1 auto-rows-[minmax(18rem,auto)] gap-4";
  const desktopGrid =
    gridClassName ??
    "xl:h-full xl:min-h-0 xl:auto-rows-fr xl:grid-cols-12 xl:grid-rows-6";

  return (
    <div className="app-shell">
      <TopNav />
      <main className="mt-4 min-h-0 flex-1 overflow-y-auto px-4 pb-4 xl:overflow-hidden xl:px-5 xl:pb-5">
        <div className={`${baseGrid} ${desktopGrid}`}>
          {children}
        </div>
      </main>
    </div>
  );
}
