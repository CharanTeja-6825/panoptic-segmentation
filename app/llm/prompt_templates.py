"""
Prompt templates for LLM scene reasoning.

Contains structured prompt builders that convert scene data into
natural language prompts for the Ollama LLM.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are an intelligent AI assistant integrated into a real-time \
video surveillance and scene understanding system. You analyse live camera feeds and \
provide insightful responses about the scene.

Your capabilities:
- Describe what is happening in the scene based on detection data
- Answer questions about object counts, movements, and patterns
- Generate alerts for unusual activity
- Provide time-based summaries of activity
- Help users understand scene dynamics
- Explain your answer using explicit visual evidence (objects, locations, motion,
  confidence, and recent events) in a concise, human-readable way.

Always base your answers on the provided scene data. Be concise, precise, and \
professional. When you lack sufficient data, say so clearly."""


def build_scene_context(scene_summary: Dict[str, Any]) -> str:
    """Convert a scene summary dict into a natural language context block.

    Args:
        scene_summary: Output from SceneMemory.get_scene_summary().

    Returns:
        Formatted context string.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"[Current Time: {now_str}]"]

    # Active objects
    active = scene_summary.get("active_objects", 0)
    counts = scene_summary.get("counts_by_class", {})
    if counts:
        obj_parts = [f"{cnt} {cls}" for cls, cnt in sorted(counts.items())]
        parts.append(f"Active objects ({active} total): {', '.join(obj_parts)}")
    else:
        parts.append("No objects currently in scene.")

    # Totals
    total = scene_summary.get("total_unique_objects_seen", 0)
    class_totals = scene_summary.get("class_totals", {})
    if class_totals:
        total_parts = [
            f"{cnt} {cls}" for cls, cnt in sorted(class_totals.items())
        ]
        parts.append(
            f"Total unique objects seen: {total} ({', '.join(total_parts)})"
        )

    # Recent events
    events = scene_summary.get("recent_events", [])
    if events:
        parts.append(f"\nRecent events ({len(events)}):")
        for evt in events[-10:]:
            ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S")
            parts.append(f"  [{ts}] {evt['description']}")

    return "\n".join(parts)


def build_chat_messages(
    user_query: str,
    scene_summary: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the full message list for an LLM chat request.

    Args:
        user_query: The user's question.
        scene_summary: Current scene data.
        chat_history: Previous conversation turns.

    Returns:
        List of message dicts for the Ollama chat API.
    """
    context = build_scene_context(scene_summary)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Current scene data:\n{context}",
        },
    ]

    # Append prior conversation history (keep limited for context window)
    if chat_history:
        for msg in chat_history[-10:]:
            messages.append(msg)

    messages.append({"role": "user", "content": user_query})
    return messages


def build_commentary_prompt(scene_summary: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a prompt for auto-generating scene commentary.

    Args:
        scene_summary: Current scene data.

    Returns:
        Message list for commentary generation.
    """
    context = build_scene_context(scene_summary)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Based on the following scene data, provide a brief 1-2 sentence "
                f"natural language description of what is happening:\n\n{context}"
            ),
        },
    ]


def build_alert_prompt(
    event: Dict[str, Any],
    scene_summary: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build a prompt to assess if an event warrants an alert.

    Args:
        event: The event that triggered the check.
        scene_summary: Current scene context.

    Returns:
        Message list for alert assessment.
    """
    context = build_scene_context(scene_summary)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"A new event occurred: {event.get('description', 'Unknown event')}\n\n"
                f"Current scene:\n{context}\n\n"
                f"Is this event noteworthy or does it require attention? "
                f"Reply with a brief assessment in one sentence."
            ),
        },
    ]


def build_video_summary_prompt(
    events: List[Dict[str, Any]],
    duration_seconds: float,
) -> List[Dict[str, str]]:
    """Build a prompt for summarising a processed video.

    Args:
        events: List of events from the video.
        duration_seconds: Total video duration.

    Returns:
        Message list for video summary.
    """
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    event_lines = []
    for evt in events[-30:]:
        ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S")
        event_lines.append(f"  [{ts}] {evt['description']}")

    events_text = "\n".join(event_lines) if event_lines else "  No events recorded."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Summarise the following video analysis results:\n"
                f"Duration: {mins}m {secs}s\n"
                f"Events ({len(events)} total):\n{events_text}\n\n"
                f"Provide a concise paragraph summarising the key activity."
            ),
        },
    ]


def build_video_chat_messages(
    user_query: str,
    video_analysis: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build prompt messages for Q&A on a processed video analysis artifact."""
    video = video_analysis.get("video", {})
    summary = video_analysis.get("summary", {})
    events = video_analysis.get("events", [])
    keyframes = video_analysis.get("keyframes", [])

    duration = float(video.get("duration_seconds", 0.0) or 0.0)
    mins = int(duration // 60)
    secs = int(duration % 60)

    event_lines: List[str] = []
    for evt in events[:80]:
        ts = float(evt.get("timestamp_seconds", 0.0) or 0.0)
        event_lines.append(
            f"  [t={ts:.2f}s] {evt.get('event_type', 'event')} | "
            f"{evt.get('class_name', 'unknown')} | track={evt.get('track_id', '?')}"
        )

    keyframe_lines: List[str] = []
    for kf in keyframes[:30]:
        keyframe_lines.append(
            f"  frame={kf.get('frame_index')} t={kf.get('timestamp_seconds')}s path={kf.get('path')}"
        )

    unique_by_class = summary.get("unique_tracks_by_class", {})
    detections_by_class = summary.get("frame_detections_by_class", {})

    context = (
        "Processed video analysis context:\n"
        f"- Duration: {mins}m {secs}s\n"
        f"- Total frames: {video.get('total_frames', 0)}\n"
        f"- Frames processed: {video.get('frames_processed', 0)}\n"
        f"- Unique tracks by class: {unique_by_class}\n"
        f"- Frame detections by class: {detections_by_class}\n"
        f"- Event count: {summary.get('events_count', 0)}\n"
        f"- Keyframe count: {summary.get('keyframes_count', 0)}\n\n"
        f"Sampled keyframes ({min(len(keyframes), 30)} shown):\n"
        f"{chr(10).join(keyframe_lines) if keyframe_lines else '  none'}\n\n"
        f"Events ({min(len(events), 80)} shown):\n"
        f"{chr(10).join(event_lines) if event_lines else '  none'}\n"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "You are answering questions about a fully processed video. "
                "Ground answers in the structured analysis and sampled keyframes metadata. "
                "If data is insufficient, say what is missing."
            ),
        },
        {"role": "system", "content": context},
        {"role": "user", "content": user_query},
    ]
