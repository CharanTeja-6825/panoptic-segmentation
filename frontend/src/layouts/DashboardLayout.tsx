import { ReactNode } from "react";
import { useStore } from "../store/useStore";
import TopNav from "../components/TopNav";

interface DashboardLayoutProps {
  children: ReactNode;
  gridClassName?: string;
}

export default function DashboardLayout({ children, gridClassName }: DashboardLayoutProps) {
  const darkMode = useStore((s) => s.darkMode);
  const baseGrid =
    "grid min-h-full grid-cols-1 auto-rows-[minmax(20rem,auto)] gap-3 md:gap-4";
  const desktopGrid =
    gridClassName ??
    "xl:h-full xl:min-h-0 xl:auto-rows-fr xl:grid-cols-12 xl:grid-rows-6";

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="app-shell">
        <TopNav />
        <main className="mt-3 min-h-0 flex-1 overflow-y-auto xl:overflow-hidden">
          <div className={`${baseGrid} ${desktopGrid}`}>
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
