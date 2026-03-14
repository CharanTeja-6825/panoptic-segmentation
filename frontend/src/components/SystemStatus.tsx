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
    { key: "camera", label: "Cam" },
    { key: "llm", label: "LLM" },
    { key: "websocket", label: "WS" },
  ];

  return (
    <div className="hidden items-center gap-1.5 sm:flex">
      {services.map((service) => {
        const ok = systemHealth[service.key];
        return (
          <div
            key={service.label}
            className="status-pill status-pill-muted"
            title={`${service.label}: ${ok ? "Connected" : "Disconnected"}`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full transition-colors ${
                ok
                  ? "bg-emerald-400"
                  : "bg-slate-500"
              }`}
              style={ok ? { boxShadow: "0 0 6px rgba(52,211,153,0.5)" } : undefined}
            />
            <span>{service.label}</span>
          </div>
        );
      })}
    </div>
  );
}
