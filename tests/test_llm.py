"""
Tests for the LLM module (prompt templates and Ollama client).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.prompt_templates import (
    SYSTEM_PROMPT,
    build_scene_context,
    build_chat_messages,
    build_commentary_prompt,
    build_alert_prompt,
    build_video_summary_prompt,
    build_video_chat_messages,
)
from app.llm.ollama_client import OllamaClient, LLMClient


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------

class TestBuildSceneContext:

    def test_empty_scene(self):
        summary = {
            "active_objects": 0,
            "counts_by_class": {},
            "total_unique_objects_seen": 0,
            "class_totals": {},
            "recent_events": [],
        }
        context = build_scene_context(summary)
        # Updated: prompt templates now use compact format
        assert "no objects" in context.lower() or "0" in context

    def test_with_objects(self):
        summary = {
            "active_objects": 3,
            "counts_by_class": {"person": 2, "car": 1},
            "total_unique_objects_seen": 5,
            "class_totals": {"person": 3, "car": 2},
            "recent_events": [],
        }
        context = build_scene_context(summary)
        assert "person" in context.lower()
        assert "car" in context.lower()

    def test_with_events(self):
        summary = {
            "active_objects": 1,
            "counts_by_class": {"person": 1},
            "total_unique_objects_seen": 1,
            "class_totals": {"person": 1},
            "recent_events": [
                {
                    "timestamp": time.time(),
                    "event_type": "entry",
                    "description": "person entered the scene (track 1)",
                    "objects_involved": [1],
                    "metadata": {},
                }
            ],
        }
        context = build_scene_context(summary)
        # Check for event content in some form
        assert "person" in context.lower()


class TestBuildChatMessages:

    def test_basic_message_structure(self):
        summary = {
            "active_objects": 0,
            "counts_by_class": {},
            "total_unique_objects_seen": 0,
            "class_totals": {},
            "recent_events": [],
        }
        messages = build_chat_messages("How many people?", summary)
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "How many people?" in messages[-1]["content"]

    def test_with_chat_history(self):
        summary = {
            "active_objects": 0,
            "counts_by_class": {},
            "total_unique_objects_seen": 0,
            "class_totals": {},
            "recent_events": [],
        }
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages = build_chat_messages("What's new?", summary, history)
        assert any(m["content"] == "Hello" for m in messages)
        assert any(m["content"] == "Hi there!" for m in messages)

    def test_system_prompt_included(self):
        summary = {
            "active_objects": 0,
            "counts_by_class": {},
            "total_unique_objects_seen": 0,
            "class_totals": {},
            "recent_events": [],
        }
        messages = build_chat_messages("test", summary)
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) >= 1


class TestBuildCommentaryPrompt:

    def test_returns_messages(self):
        summary = {
            "active_objects": 1,
            "counts_by_class": {"person": 1},
            "total_unique_objects_seen": 1,
            "class_totals": {"person": 1},
            "recent_events": [],
        }
        messages = build_commentary_prompt(summary)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Updated: commentary prompt now uses compact format
        assert "person" in messages[1]["content"].lower()


class TestBuildAlertPrompt:

    def test_includes_event(self):
        event = {
            "description": "person entered the scene",
            "event_type": "entry",
        }
        summary = {
            "active_objects": 1,
            "counts_by_class": {"person": 1},
            "total_unique_objects_seen": 1,
            "class_totals": {"person": 1},
            "recent_events": [],
        }
        messages = build_alert_prompt(event, summary)
        assert any("person entered" in m["content"] for m in messages)


class TestBuildVideoSummaryPrompt:

    def test_returns_messages(self):
        events = [
            {
                "timestamp": time.time(),
                "description": "person entered the scene",
            }
        ]
        messages = build_video_summary_prompt(events, 120.0)
        assert len(messages) == 2
        assert "2m 0s" in messages[1]["content"]


class TestBuildVideoChatMessages:

    def test_includes_analysis_summary_and_query(self):
        analysis = {
            "video": {
                "duration_seconds": 125.0,
                "total_frames": 250,
                "frames_processed": 250,
            },
            "summary": {
                "unique_tracks_by_class": {"person": 4},
                "frame_detections_by_class": {"person": 120},
                "events_count": 8,
                "keyframes_count": 5,
            },
            "events": [
                {
                    "timestamp_seconds": 1.0,
                    "event_type": "entry",
                    "class_name": "person",
                    "track_id": 1,
                }
            ],
            "keyframes": [
                {"frame_index": 0, "timestamp_seconds": 0.0, "path": "outputs/x.jpg"}
            ],
        }
        messages = build_video_chat_messages("What happened?", analysis)
        assert len(messages) >= 3
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What happened?"
        # Updated: check for video context in some form
        assert any("video" in m["content"].lower() or "analysis" in m["content"].lower() for m in messages)


# ---------------------------------------------------------------------------
# OllamaClient tests (unit-level, no real server)
# ---------------------------------------------------------------------------

class TestOllamaClient:

    def test_init_defaults(self):
        client = OllamaClient()
        assert client.base_url == "http://100.91.144.84:11434"
        assert client.model == "llama3.2"
        assert client.is_available is False

    def test_custom_init(self):
        # Updated: LLMClient now uses separate connect_timeout and read_timeout
        client = LLMClient(
            base_url="http://myserver:5000",
            model="mistral",
            connect_timeout=5.0,
            read_timeout=60.0,
        )
        assert client.base_url == "http://myserver:5000"
        assert client.model == "mistral"
        assert client.connect_timeout == 5.0
        assert client.read_timeout == 60.0

    @pytest.mark.asyncio
    async def test_check_health_unreachable(self):
        client = OllamaClient(base_url="http://localhost:99999")
        result = await client.check_health()
        assert result is False
        assert client.is_available is False

    @pytest.mark.asyncio
    async def test_list_models_unreachable(self):
        client = OllamaClient(base_url="http://localhost:99999")
        models = await client.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_chat_unreachable(self):
        client = OllamaClient(base_url="http://localhost:99999")
        result = await client.chat([{"role": "user", "content": "hello"}])
        # Updated: chat() now returns LLMResponse object
        assert result.error is not None
        assert "connection" in result.error.message.lower() or "connect" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_generate_unreachable(self):
        client = OllamaClient(base_url="http://localhost:99999")
        result = await client.generate("hello")
        # generate() returns a string, check for error indication
        assert "error" in result.lower() or "cannot" in result.lower()
