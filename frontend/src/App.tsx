import DashboardLayout from "./layouts/DashboardLayout";
import LiveFeed from "./components/LiveFeed";
import ChatPanel from "./components/ChatPanel";
import EventTimeline from "./components/EventTimeline";
import StatsPanel from "./components/StatsPanel";
import ControlPanel from "./components/ControlPanel";
import VideoWorkflowPanel from "./components/VideoWorkflowPanel";
import { PageKey, useStore } from "./store/useStore";

const PAGE_GRID: Record<PageKey, string> = {
  "live-ops": "xl:h-full xl:min-h-0 xl:auto-rows-fr xl:grid-cols-12 xl:grid-rows-4",
  assistant: "xl:h-full xl:min-h-0 xl:auto-rows-fr xl:grid-cols-12 xl:grid-rows-3",
  "video-studio":
    "xl:h-full xl:min-h-0 xl:auto-rows-fr xl:grid-cols-12 xl:grid-rows-3",
};

function LiveOpsPage() {
  return (
    <>
      <section className="min-h-[20rem] xl:col-span-8 xl:row-span-3">
        <LiveFeed />
      </section>

      <section className="min-h-[16rem] xl:col-span-4 xl:row-span-2">
        <StatsPanel />
      </section>

      <section className="min-h-[14rem] xl:col-span-4 xl:row-span-2">
        <ControlPanel />
      </section>

      <section className="min-h-[16rem] xl:col-span-8 xl:row-span-1">
        <EventTimeline />
      </section>
    </>
  );
}

function AssistantPage() {
  return (
    <>
      <section className="min-h-[22rem] xl:col-span-7 xl:row-span-3">
        <ChatPanel />
      </section>

      <section className="min-h-[22rem] xl:col-span-5 xl:row-span-3">
        <EventTimeline />
      </section>
    </>
  );
}

function VideoStudioPage() {
  return (
    <section className="min-h-[26rem] xl:col-span-12 xl:row-span-3">
      <VideoWorkflowPanel />
    </section>
  );
}

export default function App() {
  const currentPage = useStore((s) => s.currentPage);
  const gridClassName = PAGE_GRID[currentPage];

  const renderPage = () => {
    switch (currentPage) {
      case "assistant":
        return <AssistantPage />;
      case "video-studio":
        return <VideoStudioPage />;
      case "live-ops":
      default:
        return <LiveOpsPage />;
    }
  };

  return (
    <DashboardLayout gridClassName={gridClassName}>{renderPage()}</DashboardLayout>
  );
}
