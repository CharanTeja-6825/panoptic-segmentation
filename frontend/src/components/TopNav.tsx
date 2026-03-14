import { PageKey, useStore } from "../store/useStore";
import SystemStatus from "./SystemStatus";

export default function TopNav() {
  const darkMode = useStore((s) => s.darkMode);
  const toggleDarkMode = useStore((s) => s.toggleDarkMode);
  const cameraActive = useStore((s) => s.cameraActive);
  const recording = useStore((s) => s.recording);
  const currentPage = useStore((s) => s.currentPage);
  const setCurrentPage = useStore((s) => s.setCurrentPage);
  const now = new Date();
  const navItems: Array<{ key: PageKey; label: string }> = [
    { key: "live-ops", label: "Live Ops" },
    { key: "assistant", label: "Assistant" },
    { key: "video-studio", label: "Video Studio" },
  ];

  return (
    <header className="rounded-2xl border border-slate-700/60 bg-slate-900/70 px-4 py-3 shadow-lg shadow-slate-950/40 backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 shadow-lg shadow-indigo-500/20">
            <svg
              className="h-5 w-5 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
              />
            </svg>
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold tracking-tight text-white">
              Scene Understanding Console
            </h1>
            <p className="text-xs text-slate-400">
              Real-time YOLO segmentation + explainable AI assistant
            </p>
          </div>
        </div>

        <div className="hidden text-right lg:block">
          <p className="text-xs uppercase tracking-wider text-slate-500">Session</p>
          <p className="text-sm font-medium text-slate-200">
            {now.toLocaleDateString()} · {now.toLocaleTimeString()}
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          <SystemStatus />
          {cameraActive && (
            <span className="status-pill status-pill-success">
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-300" />
              Live Camera
            </span>
          )}
          {recording && (
            <span className="status-pill border-rose-500/30 bg-rose-500/10 text-rose-300">
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-rose-300" />
              Recording
            </span>
          )}
          <button
            onClick={toggleDarkMode}
            className="rounded-lg border border-slate-700 bg-slate-800/80 p-2 text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-700/80 hover:text-white"
            title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {darkMode ? (
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                />
              </svg>
            ) : (
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
      <nav className="mt-3 flex flex-wrap items-center gap-2">
        {navItems.map((item) => {
          const active = currentPage === item.key;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setCurrentPage(item.key)}
              aria-current={active ? "page" : undefined}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                active
                  ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/30"
                  : "border border-slate-700 bg-slate-800/80 text-slate-200 hover:border-indigo-500/50 hover:bg-indigo-500/10 hover:text-white"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-400 lg:hidden">
        <span className="rounded-md bg-slate-800 px-2 py-1">
          {now.toLocaleDateString()}
        </span>
        <span className="rounded-md bg-slate-800 px-2 py-1">
          {now.toLocaleTimeString()}
        </span>
      </div>
    </header>
  );
}
