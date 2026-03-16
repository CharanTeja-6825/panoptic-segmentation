import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';

import { Sidebar } from '@/components/layout/Sidebar';
import { TopNav } from '@/components/layout/TopNav';
import { useSceneEventsSocket } from '@/hooks/useSceneEventsSocket';
import { useUiStore } from '@/store/uiStore';

export const AppLayout = () => {
  const { theme, sidebarOpen } = useUiStore();
  useSceneEventsSocket();

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      {sidebarOpen ? <Sidebar /> : null}
      <main className="flex min-h-screen min-w-0 flex-1 flex-col">
        <TopNav />
        <div className="flex-1 overflow-auto p-4 md:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
};
