import DashboardLayout from "./layouts/DashboardLayout";
import LiveFeed from "./components/LiveFeed";
import ChatPanel from "./components/ChatPanel";
import EventTimeline from "./components/EventTimeline";
import StatsPanel from "./components/StatsPanel";
import ControlPanel from "./components/ControlPanel";

export default function App() {
  return (
    <DashboardLayout>
      {/* Top Left - Live Feed */}
      <LiveFeed />

      {/* Top Right - Chat */}
      <ChatPanel />

      {/* Bottom Left - Events + Controls */}
      <div className="flex flex-col gap-4 overflow-hidden">
        <div className="flex-1 overflow-hidden">
          <EventTimeline />
        </div>
        <ControlPanel />
      </div>

      {/* Bottom Right - Stats */}
      <StatsPanel />
    </DashboardLayout>
  );
}
