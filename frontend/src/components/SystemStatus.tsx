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
        if (mounted) setSystemHealth({ llm: status.connected });
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

  const services = [
    { label: "API", ok: systemHealth.api },
    { label: "Camera", ok: systemHealth.camera },
    { label: "LLM", ok: systemHealth.llm },
    { label: "WS", ok: systemHealth.websocket },
  ];

  return (
    <div className="flex items-center gap-3">
      {services.map((s) => (
        <div key={s.label} className="flex items-center gap-1.5" title={`${s.label}: ${s.ok ? "Connected" : "Disconnected"}`}>
          <span
            className={`h-2 w-2 rounded-full ${
              s.ok
                ? "bg-emerald-400 shadow-sm shadow-emerald-400/50"
                : "bg-slate-500"
            }`}
          />
          <span className="text-xs text-slate-400">{s.label}</span>
        </div>
      ))}
    </div>
  );
}
