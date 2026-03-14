import { SceneEvent, SceneObject } from "../store/useStore";

function asFiniteNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

export function normalizeSceneObject(raw: Record<string, unknown>): SceneObject {
  const bboxRaw = Array.isArray(raw.bbox) ? raw.bbox : [];
  const x1 = clamp(asFiniteNumber(bboxRaw[0]), 0, 100);
  const y1 = clamp(asFiniteNumber(bboxRaw[1]), 0, 100);
  const x2 = clamp(asFiniteNumber(bboxRaw[2]), x1, 100);
  const y2 = clamp(asFiniteNumber(bboxRaw[3]), y1, 100);

  return {
    id: String(raw.id ?? `obj-${Date.now()}`),
    label: String(raw.label ?? "unknown"),
    confidence: clamp(asFiniteNumber(raw.confidence, 0), 0, 1),
    bbox: [x1, y1, x2, y2],
  };
}

export function normalizeSceneEvent(raw: Record<string, unknown>): SceneEvent {
  const eventType = String(raw.type ?? raw.event_type ?? "info");
  const mappedType: SceneEvent["type"] =
    eventType === "entry" || eventType === "exit" ? "detection" : "info";
  const type = (
    ["detection", "alert", "warning", "info", "system"].includes(eventType)
      ? eventType
      : mappedType
  ) as SceneEvent["type"];

  const timeValue = raw.timestamp;
  const timestamp =
    typeof timeValue === "number"
      ? new Date(timeValue * 1000).toISOString()
      : String(timeValue ?? new Date().toISOString());

  const severity = String(raw.severity ?? "low");
  const safeSeverity: SceneEvent["severity"] =
    severity === "high" || severity === "medium" ? severity : "low";

  return {
    id: String(raw.id ?? `evt-${Date.now()}`),
    type,
    message: String(raw.message ?? raw.description ?? "No details provided."),
    timestamp,
    severity: safeSeverity,
  };
}

export function summarizeObjectCounts(objects: SceneObject[]): Record<string, number> {
  return objects.reduce<Record<string, number>>((acc, obj) => {
    acc[obj.label] = (acc[obj.label] ?? 0) + 1;
    return acc;
  }, {});
}
