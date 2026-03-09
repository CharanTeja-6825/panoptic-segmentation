"""
Live commentary engine.

Auto-generates scene descriptions, priority alerts, and human-readable
summaries by combining scene memory with the Ollama LLM.
"""

import asyncio
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from app.llm.ollama_client import OllamaClient
from app.llm.prompt_templates import (
    build_commentary_prompt,
    build_alert_prompt,
)
from app.memory.scene_memory import SceneMemory

logger = logging.getLogger(__name__)


@dataclass
class Commentary:
    """A generated commentary entry."""

    timestamp: float
    text: str
    commentary_type: str = "description"  # description | alert | summary
    priority: str = "normal"  # low | normal | high | critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "text": self.text,
            "type": self.commentary_type,
            "priority": self.priority,
        }


class CommentaryEngine:
    """Generates live scene commentary using LLM.

    Args:
        ollama_client: The Ollama LLM client.
        scene_memory: The scene memory instance.
        commentary_interval: Minimum seconds between auto-commentaries.
        max_history: Maximum commentary entries to retain.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        scene_memory: SceneMemory,
        commentary_interval: float = 30.0,
        max_history: int = 100,
    ) -> None:
        self._client = ollama_client
        self._memory = scene_memory
        self._interval = commentary_interval
        self._lock = threading.Lock()
        self._history: Deque[Commentary] = deque(maxlen=max_history)
        self._last_commentary_time: float = 0.0
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    async def maybe_generate_commentary(self) -> Optional[Commentary]:
        """Generate auto-commentary if enough time has elapsed.

        Returns:
            A Commentary object if generated, else None.
        """
        if not self._enabled or not self._client.is_available:
            return None

        now = time.time()
        if now - self._last_commentary_time < self._interval:
            return None

        scene_summary = self._memory.get_scene_summary()
        if scene_summary.get("active_objects", 0) == 0:
            return None

        try:
            messages = build_commentary_prompt(scene_summary)
            text = await self._client.chat(messages, temperature=0.5)
            if text:
                commentary = Commentary(
                    timestamp=now,
                    text=text.strip(),
                    commentary_type="description",
                )
                with self._lock:
                    self._history.append(commentary)
                self._last_commentary_time = now
                logger.debug("Generated commentary: %s", text[:80])
                return commentary
        except Exception as exc:
            logger.warning("Commentary generation failed: %s", exc)

        return None

    async def assess_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[Commentary]:
        """Assess whether an event is noteworthy.

        Args:
            event: Scene event dict.

        Returns:
            Alert Commentary if noteworthy, else None.
        """
        if not self._enabled or not self._client.is_available:
            return None

        scene_summary = self._memory.get_scene_summary()
        try:
            messages = build_alert_prompt(event, scene_summary)
            text = await self._client.chat(messages, temperature=0.3)
            if text:
                commentary = Commentary(
                    timestamp=time.time(),
                    text=text.strip(),
                    commentary_type="alert",
                    priority="high",
                )
                with self._lock:
                    self._history.append(commentary)
                return commentary
        except Exception as exc:
            logger.warning("Event assessment failed: %s", exc)

        return None

    def get_recent_commentary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent commentary entries."""
        with self._lock:
            items = list(self._history)
        items.reverse()
        return [c.to_dict() for c in items[:limit]]

    def add_manual_commentary(self, text: str, priority: str = "normal") -> Commentary:
        """Add a non-LLM commentary entry (e.g. system messages)."""
        commentary = Commentary(
            timestamp=time.time(),
            text=text,
            commentary_type="system",
            priority=priority,
        )
        with self._lock:
            self._history.append(commentary)
        return commentary

    def reset(self) -> None:
        """Clear commentary history."""
        with self._lock:
            self._history.clear()
            self._last_commentary_time = 0.0
        logger.info("Commentary engine reset")
