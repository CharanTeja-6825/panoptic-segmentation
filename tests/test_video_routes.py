"""
Tests for processed-video analysis and chat routes.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.config import config
from app.routes import video_routes


@pytest.fixture(autouse=True)
def _isolate_jobs():
    original_jobs = dict(video_routes._jobs)
    video_routes._jobs.clear()
    yield
    video_routes._jobs.clear()
    video_routes._jobs.update(original_jobs)


def _response_json(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def _make_request_with_client(client) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(ollama_client=client))
    scope = {"type": "http", "method": "POST", "path": "/api/video-chat/test", "headers": [], "app": app}
    return Request(scope)


@pytest.mark.asyncio
async def test_job_status_includes_analysis_url():
    video_routes._jobs["job-1"] = {
        "job_id": "job-1",
        "status": "done",
        "progress": 10,
        "total_frames": 10,
        "fps": 20.0,
        "error": None,
        "analysis_path": "/tmp/example_analysis.json",
        "stats": {},
    }
    response = await video_routes.job_status("job-1")
    body = _response_json(response)
    assert body["analysis_url"] == "/api/video-analysis/job-1"


@pytest.mark.asyncio
async def test_video_analysis_returns_keyframe_urls(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    keyframe_dir = output_dir / "job-1_keyframes"
    keyframe_dir.mkdir()
    keyframe_path = keyframe_dir / "frame_000000.jpg"
    keyframe_path.write_bytes(b"fake-jpg")
    analysis_path = output_dir / "job-1_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "video": {"total_frames": 12},
                "summary": {"events_count": 1, "keyframes_count": 1},
                "events": [],
                "keyframes": [{"path": str(keyframe_path), "frame_index": 0}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "output_dir", str(output_dir))
    video_routes._jobs["job-1"] = {
        "job_id": "job-1",
        "status": "done",
        "analysis_path": str(analysis_path),
    }

    response = await video_routes.video_analysis("job-1")
    body = _response_json(response)
    assert body["analysis"]["keyframes"][0]["url"].startswith("/outputs/")


@pytest.mark.asyncio
async def test_video_chat_uses_keyframes_and_vision_model(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    keyframe = output_dir / "frame_000001.jpg"
    keyframe.write_bytes(b"keyframe-bytes")
    analysis_path = output_dir / "job-1_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "video": {"duration_seconds": 10.0, "total_frames": 100, "frames_processed": 100},
                "summary": {
                    "unique_tracks_by_class": {"person": 1},
                    "frame_detections_by_class": {"person": 10},
                    "events_count": 1,
                    "keyframes_count": 1,
                },
                "events": [],
                "keyframes": [{"path": str(keyframe), "frame_index": 1, "timestamp_seconds": 0.1}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "output_dir", str(output_dir))
    video_routes._jobs["job-1"] = {
        "job_id": "job-1",
        "status": "done",
        "analysis_path": str(analysis_path),
    }

    client = MagicMock()
    client.model = "llama3.2"
    client.encode_image_bytes.return_value = "encoded-image"
    client.chat = AsyncMock(return_value="Observed one person.")
    request = _make_request_with_client(client)

    response = await video_routes.video_chat(
        "job-1",
        video_routes.VideoChatRequest(message="What happened?", max_keyframes=3),
        request,
    )
    body = _response_json(response)
    assert body["reply"] == "Observed one person."
    assert body["used_keyframes"] == 1
    assert body["model"] == config.ollama_vision_model
    assert client.chat.await_count == 1


@pytest.mark.asyncio
async def test_video_chat_rejects_empty_message():
    with pytest.raises(HTTPException) as exc:
        await video_routes.video_chat(
            "job-1",
            video_routes.VideoChatRequest(message="   "),
            _make_request_with_client(MagicMock()),
        )
    assert exc.value.status_code == 400
