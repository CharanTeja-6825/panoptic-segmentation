import DashboardLayout from "./layouts/DashboardLayout";
import LiveFeed from "./components/LiveFeed";
import ChatPanel from "./components/ChatPanel";
import EventTimeline from "./components/EventTimeline";
import StatsPanel from "./components/StatsPanel";
import ControlPanel from "./components/ControlPanel";

export default function App() {
  return (
    <DashboardLayout>
      <section className="min-h-[24rem] xl:min-h-0 xl:col-span-7 xl:row-span-4">
        <LiveFeed />
      </section>

      <section className="min-h-[24rem] xl:min-h-0 xl:col-span-5 xl:row-span-4">
        <ChatPanel />
      </section>

      <section className="flex min-h-[24rem] flex-col gap-3 md:gap-4 xl:min-h-0 xl:col-span-7 xl:row-span-2">
        <div className="min-h-0 flex-1">
          <EventTimeline />
        </div>
        <ControlPanel />
      </section>

      <section className="min-h-[24rem] xl:min-h-0 xl:col-span-5 xl:row-span-2">
        <StatsPanel />
      </section>
    </DashboardLayout>
  );
}
