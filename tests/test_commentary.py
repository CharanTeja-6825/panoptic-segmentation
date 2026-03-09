"""
Tests for the commentary engine.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.commentary_engine import CommentaryEngine, Commentary
from app.llm.ollama_client import OllamaClient
from app.memory.scene_memory import SceneMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTrackedObject:
    """Minimal stand-in for TrackedObject."""

    def __init__(self, track_id=1, class_name="person", confidence=0.9,
                 bbox=(100, 100, 200, 200)):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.centroid = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


def _make_engine(
    available: bool = True,
    chat_response: str = "Scene description.",
) -> tuple:
    """Create a CommentaryEngine with mocked dependencies."""
    client = MagicMock(spec=OllamaClient)
    client.is_available = available
    client.chat = AsyncMock(return_value=chat_response)
    memory = SceneMemory()
    engine = CommentaryEngine(
        ollama_client=client,
        scene_memory=memory,
        commentary_interval=0.0,  # no delay for tests
    )
    return engine, client, memory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCommentary:

    def test_to_dict(self):
        c = Commentary(
            timestamp=1234567890.0,
            text="Two people in view",
            commentary_type="description",
            priority="normal",
        )
        d = c.to_dict()
        assert d["text"] == "Two people in view"
        assert d["type"] == "description"
        assert d["priority"] == "normal"


class TestCommentaryEngine:

    def test_initial_state(self):
        engine, _, _ = _make_engine()
        assert engine.enabled is True
        assert engine.get_recent_commentary() == []

    def test_enable_disable(self):
        engine, _, _ = _make_engine()
        engine.enabled = False
        assert engine.enabled is False

    @pytest.mark.asyncio
    async def test_no_commentary_when_disabled(self):
        engine, _, memory = _make_engine()
        engine.enabled = False
        memory.update_scene([FakeTrackedObject()])
        result = await engine.maybe_generate_commentary()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_commentary_when_no_objects(self):
        engine, _, _ = _make_engine()
        result = await engine.maybe_generate_commentary()
        assert result is None

    @pytest.mark.asyncio
    async def test_generates_commentary(self):
        engine, client, memory = _make_engine()
        memory.update_scene([FakeTrackedObject()])
        result = await engine.maybe_generate_commentary()
        assert result is not None
        assert result.text == "Scene description."
        assert result.commentary_type == "description"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_commentary_stored_in_history(self):
        engine, _, memory = _make_engine()
        memory.update_scene([FakeTrackedObject()])
        await engine.maybe_generate_commentary()
        history = engine.get_recent_commentary()
        assert len(history) == 1
        assert history[0]["text"] == "Scene description."

    @pytest.mark.asyncio
    async def test_no_commentary_when_llm_unavailable(self):
        engine, _, memory = _make_engine(available=False)
        memory.update_scene([FakeTrackedObject()])
        result = await engine.maybe_generate_commentary()
        assert result is None

    @pytest.mark.asyncio
    async def test_assess_event(self):
        engine, client, memory = _make_engine(
            chat_response="This is a routine entry."
        )
        memory.update_scene([FakeTrackedObject()])
        event = {
            "description": "person entered the scene",
            "event_type": "entry",
        }
        result = await engine.assess_event(event)
        assert result is not None
        assert result.commentary_type == "alert"
        assert result.priority == "high"

    def test_manual_commentary(self):
        engine, _, _ = _make_engine()
        result = engine.add_manual_commentary("System started", priority="low")
        assert result.commentary_type == "system"
        assert result.priority == "low"
        history = engine.get_recent_commentary()
        assert len(history) == 1

    def test_reset(self):
        engine, _, _ = _make_engine()
        engine.add_manual_commentary("test")
        engine.reset()
        assert engine.get_recent_commentary() == []
