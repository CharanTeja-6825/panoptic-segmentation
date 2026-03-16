import { NavLink } from 'react-router-dom';

import { ROUTES } from '@/constants/routes';
import { cn } from '@/utils/classNames';

const links = [
  { to: ROUTES.dashboard, label: 'Dashboard' },
  { to: ROUTES.camera, label: 'Camera' },
  { to: ROUTES.video, label: 'Video' },
  { to: ROUTES.analytics, label: 'Analytics' },
  { to: ROUTES.scene, label: 'Scene' },
  { to: ROUTES.chat, label: 'Chat' },
];

export const Sidebar = () => (
  <aside className="h-full w-64 shrink-0 border-r border-slate-800 bg-slate-950 p-4">
    <div className="mb-6">
      <h1 className="text-lg font-semibold text-slate-100">Panoptic Console</h1>
      <p className="text-xs text-slate-400">Realtime scene intelligence</p>
    </div>
    <nav className="space-y-1">
      {links.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className={({ isActive }) =>
            cn(
              'block rounded-md px-3 py-2 text-sm transition',
              isActive ? 'bg-indigo-500/20 text-indigo-200' : 'text-slate-300 hover:bg-slate-800 hover:text-slate-100',
            )
          }
          end={item.to === ROUTES.dashboard}
        >
          {item.label}
        </NavLink>
      ))}
    </nav>
  </aside>
);
