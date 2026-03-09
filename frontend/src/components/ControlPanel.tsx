import { useStore } from "../store/useStore";
import { startCamera, stopCamera } from "../services/api";

export default function ControlPanel() {
  const cameraActive = useStore((s) => s.cameraActive);
  const setCameraActive = useStore((s) => s.setCameraActive);
  const setSystemHealth = useStore((s) => s.setSystemHealth);
  const detectionSensitivity = useStore((s) => s.detectionSensitivity);
  const setDetectionSensitivity = useStore((s) => s.setDetectionSensitivity);
  const recording = useStore((s) => s.recording);
  const setRecording = useStore((s) => s.setRecording);

  const handleCameraToggle = async () => {
    try {
      if (cameraActive) {
        await stopCamera();
        setCameraActive(false);
        setSystemHealth({ camera: false });
      } else {
        await startCamera();
        setCameraActive(true);
        setSystemHealth({ camera: true });
      }
    } catch {
      // Handle error silently
    }
  };

  const handleRecordToggle = () => {
    setRecording(!recording);
  };

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-slate-700/50 bg-slate-800 p-4 shadow-lg">
      <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400">
        Controls
      </h3>

      {/* Camera Toggle */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-300">Camera</span>
        <button
          onClick={handleCameraToggle}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            cameraActive ? "bg-emerald-600" : "bg-slate-600"
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
              cameraActive ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      {/* Recording Toggle */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-300">Recording</span>
        <button
          onClick={handleRecordToggle}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            recording ? "bg-rose-600" : "bg-slate-600"
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
              recording ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      {/* Detection Sensitivity */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm text-slate-300">Sensitivity</span>
          <span className="text-xs font-semibold text-indigo-400">
            {detectionSensitivity}%
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={detectionSensitivity}
          onChange={(e) => setDetectionSensitivity(Number(e.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-indigo-500"
        />
      </div>
    </div>
  );
}
