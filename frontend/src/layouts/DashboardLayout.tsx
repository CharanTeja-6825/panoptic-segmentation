import { ReactNode } from "react";
import TopNav from "../components/TopNav";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="flex h-screen flex-col bg-slate-900 text-slate-100">
      <TopNav />
      <main className="flex-1 overflow-hidden p-4">
        <div className="grid h-full grid-cols-1 gap-4 lg:grid-cols-2 lg:grid-rows-2">
          {children}
        </div>
      </main>
    </div>
  );
}
