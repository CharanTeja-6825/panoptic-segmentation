import { useEffect } from "react";
import { useStore } from "../store/useStore";
import { getHealth, getLLMStatus } from "../services/api";

export default function SystemStatus() {
  const systemHealth = useStore((s) => s.systemHealth);
  const setSystemHealth = useStore((s) => s.setSystemHealth);

  useEffect(() => {
    let mounted = true;

    const checkHealth = async () => {
      try {
        await getHealth();
        if (mounted) setSystemHealth({ api: true });
      } catch {
        if (mounted) setSystemHealth({ api: false });
      }

      try {
        const status = await getLLMStatus();
        if (mounted) setSystemHealth({ llm: Boolean(status.available) });
      } catch {
        if (mounted) setSystemHealth({ llm: false });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [setSystemHealth]);

  const services: Array<{ key: keyof typeof systemHealth; label: string }> = [
    { key: "api", label: "API" },
    { key: "camera", label: "Camera" },
    { key: "llm", label: "LLM" },
    { key: "websocket", label: "WebSocket" },
  ];

  return (
    <div className="flex flex-wrap items-center gap-2">
      {services.map((service) => {
        const ok = systemHealth[service.key];
        return (
          <div
            key={service.label}
            className="status-pill status-pill-muted"
            title={`${service.label}: ${ok ? "Connected" : "Disconnected"}`}
          >
          <span
              className={`h-2 w-2 rounded-full ${
                ok
                ? "bg-emerald-400 shadow-sm shadow-emerald-400/50"
                : "bg-slate-500"
              }`}
            />
            <span className={ok ? "text-emerald-200" : "text-slate-300"}>
              {service.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
