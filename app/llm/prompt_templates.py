"""
Prompt templates for LLM scene reasoning.

Contains structured prompt builders that convert scene data into
natural language prompts for the Ollama LLM.

Optimized for low-resource hardware with context compaction and token budgeting.
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Token budget limits for low-resource environments
MAX_CONTEXT_EVENTS = 10
MAX_CHAT_HISTORY = 6
MAX_VIDEO_EVENTS = 50
MAX_KEYFRAMES = 20
ESTIMATED_TOKENS_PER_CHAR = 0.25
MAX_CONTEXT_TOKENS = 2000

SYSTEM_PROMPT = """You are an intelligent AI assistant integrated into a real-time \
video surveillance and scene understanding system. You analyse live camera feeds and \
provide insightful responses about the scene.

Your capabilities:
- Describe what is happening in the scene based on detection data
- Answer questions about object counts, movements, and patterns
- Generate alerts for unusual activity
- Provide time-based summaries of activity
- Help users understand scene dynamics

Always base your answers on the provided scene data. Be concise and precise. \
When you lack sufficient data, say so clearly."""

# Compact system prompt for vision queries (saves tokens)
VISION_SYSTEM_PROMPT = """You are an AI vision assistant. Describe what you see \
in the image with scene context. Be concise and factual."""


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return int(len(text) * ESTIMATED_TOKENS_PER_CHAR)


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    max_chars = int(max_tokens / ESTIMATED_TOKENS_PER_CHAR)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def build_scene_context(
    scene_summary: Dict[str, Any],
    max_events: int = MAX_CONTEXT_EVENTS,
    compact: bool = False,
) -> str:
    """
    Convert a scene summary dict into a natural language context block.

    Args:
        scene_summary: Output from SceneMemory.get_scene_summary().
        max_events: Maximum number of recent events to include.
        compact: If True, use minimal formatting for lower token count.

    Returns:
        Formatted context string.
    """
    now_str = datetime.now().strftime("%H:%M:%S")
    parts = []

    if not compact:
        parts.append(f"[Time: {now_str}]")

    # Active objects (compact format)
    active = scene_summary.get("active_objects", 0)
    counts = scene_summary.get("counts_by_class", {})
    if counts:
        obj_parts = [f"{cnt} {cls}" for cls, cnt in sorted(counts.items())]
        if compact:
            parts.append(f"Active: {', '.join(obj_parts)}")
        else:
            parts.append(f"Active objects ({active}): {', '.join(obj_parts)}")
    else:
        parts.append("No objects in scene.")

    # Totals (only include if significant)
    total = scene_summary.get("total_unique_objects_seen", 0)
    class_totals = scene_summary.get("class_totals", {})
    if class_totals and not compact:
        top_classes = sorted(class_totals.items(), key=lambda x: -x[1])[:5]
        total_parts = [f"{cnt} {cls}" for cls, cnt in top_classes]
        parts.append(f"Total seen: {total} ({', '.join(total_parts)})")

    # Recent events (limited)
    events = scene_summary.get("recent_events", [])
    if events:
        recent = events[-max_events:]
        if compact:
            parts.append(f"Events ({len(recent)}):")
            for evt in recent:
                ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S")
                desc = evt["description"][:60] if len(evt["description"]) > 60 else evt["description"]
                parts.append(f"  [{ts}] {desc}")
        else:
            parts.append(f"\nRecent events ({len(recent)}):")
            for evt in recent:
                ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S")
                parts.append(f"  [{ts}] {evt['description']}")

    return "\n".join(parts)


def build_compact_context(scene_summary: Dict[str, Any]) -> str:
    """Build a minimal context for simple queries (saves tokens)."""
    counts = scene_summary.get("counts_by_class", {})
    if not counts:
        return "Scene: empty"

    obj_list = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    return f"Scene: {obj_list}"


def classify_query(query: str) -> Tuple[str, bool]:
    """
    Classify query to determine if it needs LLM or can be answered from memory.

    Returns:
        Tuple of (query_type, needs_llm)
        query_type: 'count', 'status', 'history', 'describe', 'analyze'
        needs_llm: Whether LLM is required for this query
    """
    query_lower = query.lower().strip()

    # Count queries - can be answered from scene memory directly
    count_patterns = [
        r"how many",
        r"count of",
        r"number of",
        r"total .*(?:objects?|people|persons|cars?|vehicles?)",
    ]
    for pattern in count_patterns:
        if re.search(pattern, query_lower):
            return "count", False

    # Status queries - can be answered from scene memory directly
    status_patterns = [
        r"what'?s? (?:in|on) (?:the )?scene",
        r"current (?:objects?|status)",
        r"is there (?:a|any)",
        r"are there (?:any)?",
    ]
    for pattern in status_patterns:
        if re.search(pattern, query_lower):
            return "status", False

    # History queries - can be partially answered from memory
    history_patterns = [
        r"recent (?:events?|activity)",
        r"what happened",
        r"when did",
        r"last (?:\d+ )?(?:events?|minutes?|hours?)",
    ]
    for pattern in history_patterns:
        if re.search(pattern, query_lower):
            return "history", False

    # Description queries - need LLM
    describe_patterns = [
        r"describe",
        r"what do you see",
        r"what is happening",
        r"tell me about",
        r"explain",
    ]
    for pattern in describe_patterns:
        if re.search(pattern, query_lower):
            return "describe", True

    # Default: needs LLM for complex reasoning
    return "analyze", True


def build_deterministic_response(
    query_type: str,
    scene_summary: Dict[str, Any],
) -> Optional[str]:
    """
    Build a deterministic response for simple queries without LLM.

    Returns:
        Response string if query can be answered, None if LLM is needed.
    """
    counts = scene_summary.get("counts_by_class", {})
    active = scene_summary.get("active_objects", 0)
    events = scene_summary.get("recent_events", [])
    total = scene_summary.get("total_unique_objects_seen", 0)

    if query_type == "count":
        if not counts:
            return "There are currently no objects detected in the scene."
        obj_list = ", ".join(f"{v} {k}(s)" for k, v in sorted(counts.items()))
        return f"Currently in scene: {obj_list}. Total active objects: {active}."

    if query_type == "status":
        if not counts:
            return "The scene is currently empty with no detected objects."
        obj_list = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        return f"Scene status: {obj_list} detected. {total} unique objects seen total."

    if query_type == "history":
        if not events:
            return "No recent events recorded in scene memory."
        recent = events[-5:]
        lines = ["Recent events:"]
        for evt in recent:
            ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S")
            lines.append(f"  [{ts}] {evt['description']}")
        return "\n".join(lines)

    return None


def build_chat_messages(
    user_query: str,
    scene_summary: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    compact: bool = True,
) -> List[Dict[str, str]]:
    """
    Build the full message list for an LLM chat request.

    Args:
        user_query: The user's question.
        scene_summary: Current scene data.
        chat_history: Previous conversation turns.
        compact: Use compact context to save tokens.

    Returns:
        List of message dicts for the Ollama chat API.
    """
    context = build_scene_context(scene_summary, compact=compact)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Scene data:\n{context}"},
    ]

    # Append limited prior conversation history
    if chat_history:
        # Only keep last few exchanges to stay within token budget
        limited_history = chat_history[-MAX_CHAT_HISTORY:]
        for msg in limited_history:
            messages.append(msg)

    messages.append({"role": "user", "content": user_query})
    return messages


def build_vision_messages(
    user_query: str,
    scene_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Build messages for vision queries (when image is attached).

    Uses compact prompts to save tokens for image processing.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
    ]

    # Add minimal scene context if available
    if scene_summary:
        context = build_compact_context(scene_summary)
        messages.append({"role": "system", "content": context})

    messages.append({"role": "user", "content": user_query})
    return messages


def build_commentary_prompt(
    scene_summary: Dict[str, Any],
    compact: bool = True,
) -> List[Dict[str, str]]:
    """
    Build a prompt for auto-generating scene commentary.

    Args:
        scene_summary: Current scene data.
        compact: Use compact context to save tokens.

    Returns:
        Message list for commentary generation.
    """
    context = build_scene_context(scene_summary, max_events=5, compact=compact)
    return [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Brief 1-2 sentence scene description:\n{context}",
        },
    ]


def build_alert_prompt(
    event: Dict[str, Any],
    scene_summary: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Build a prompt to assess if an event warrants an alert.

    Args:
        event: The event that triggered the check.
        scene_summary: Current scene context.

    Returns:
        Message list for alert assessment.
    """
    context = build_compact_context(scene_summary)
    return [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Event: {event.get('description', 'Unknown')}\n"
                f"{context}\n"
                f"Is this noteworthy? One sentence."
            ),
        },
    ]


def build_video_summary_prompt(
    events: List[Dict[str, Any]],
    duration_seconds: float,
) -> List[Dict[str, str]]:
    """
    Build a prompt for summarising a processed video.

    Args:
        events: List of events from the video.
        duration_seconds: Total video duration.

    Returns:
        Message list for video summary.
    """
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)

    # Limit events to save tokens
    limited_events = events[-MAX_VIDEO_EVENTS:]
    event_lines = []
    for evt in limited_events:
        ts = datetime.fromtimestamp(evt["timestamp"]).strftime("%H:%M:%S")
        desc = evt["description"][:50] if len(evt["description"]) > 50 else evt["description"]
        event_lines.append(f"[{ts}] {desc}")

    events_text = "\n".join(event_lines) if event_lines else "No events."

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Video summary ({mins}m {secs}s, {len(events)} events):\n"
                f"{events_text}\n\n"
                f"Summarise key activity in one paragraph."
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

    # Limit to save tokens
    limited_events = events[:MAX_VIDEO_EVENTS]
    limited_keyframes = keyframes[:MAX_KEYFRAMES]

    event_lines: List[str] = []
    for evt in limited_events:
        ts = float(evt.get("timestamp_seconds", 0.0) or 0.0)
        event_lines.append(
            f"[t={ts:.1f}s] {evt.get('event_type', 'event')} "
            f"{evt.get('class_name', 'obj')} #{evt.get('track_id', '?')}"
        )

    keyframe_lines: List[str] = []
    for kf in limited_keyframes:
        keyframe_lines.append(
            f"f{kf.get('frame_index')} t={kf.get('timestamp_seconds')}s"
        )

    unique_by_class = summary.get("unique_tracks_by_class", {})

    context = (
        f"Video: {mins}m {secs}s, {video.get('frames_processed', 0)} frames\n"
        f"Tracks: {unique_by_class}\n"
        f"Events ({len(events)}): {'; '.join(event_lines[:10])}\n"
        f"Keyframes: {', '.join(keyframe_lines[:10])}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "Answer questions about the processed video. "
                "Ground answers in the analysis data."
            ),
        },
        {"role": "system", "content": context},
        {"role": "user", "content": user_query},
    ]
