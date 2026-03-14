import { ReactNode } from "react";
import { useStore } from "../store/useStore";
import TopNav from "../components/TopNav";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const darkMode = useStore((s) => s.darkMode);

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="app-shell">
        <TopNav />
        <main className="mt-3 min-h-0 flex-1 overflow-y-auto xl:overflow-hidden">
          <div className="grid min-h-full grid-cols-1 auto-rows-[minmax(20rem,auto)] gap-3 md:gap-4 xl:h-full xl:min-h-0 xl:auto-rows-fr xl:grid-cols-12 xl:grid-rows-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
