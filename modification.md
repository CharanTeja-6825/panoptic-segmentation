# Copilot Implementation Plan: Stabilize Local Multimodal Chat Pipeline

## Goal
Optimize chat + frame-to-LLM flow for low-resource hardware (8GB RAM, i3 CPU), prevent overload errors, and keep latency around 4–5 seconds for single-user usage.

## Current Problem Summary
- Frontend captures/sends too many frames for chat context.
- Backend forwards too much image data to Ollama (`llava-phi3`, `llava:7b`) causing server strain.
- Errors are generic and not observable enough.
- No clear backpressure, queueing, or rate control between frontend and backend.

## Required Architecture Changes

### 1) Frontend: Add frame throttling + dedup + in-flight lock
Update React frame sending logic to:
- Send at **max 0.5 FPS** for LLM context (`CHAT_FRAME_INTERVAL_MS=2000`).
- Skip frames if:
  - frame hash is near-identical to previous frame (simple perceptual hash or pixel diff threshold),
  - previous chat-frame request is still in-flight.
- Compress image before sending:
  - Resize to max width `512`
  - JPEG quality `55–65`
- Never send raw continuous stream frames to LLM endpoint.

**Expected:** 60–90% drop in LLM image calls.

---

### 2) Backend: Introduce bounded async queue for LLM tasks
In `app/routes/chat_routes.py` (or equivalent), implement:
- Global bounded queue: `asyncio.Queue(maxsize=2)` for chat image tasks.
- Worker coroutine (1 worker only on i3 machine) consumes queue and calls Ollama.
- If queue full:
  - return HTTP `429` with message `"LLM busy, try again shortly"`.

Do not process multiple image-heavy chat requests concurrently.

---

### 3) Backend: Add request timeout, cancellation, and fallback model logic
In `app/llm/ollama_client.py`:
- Set strict timeout:
  - connect timeout: 5s
  - read timeout: 45s
- If primary model `llava:7b` fails or exceeds latency budget:
  - fallback to lighter model (configurable), e.g. `llava-phi3` or text-only model.
- Return structured errors instead of generic `"error"`:
  - `OLLAMA_TIMEOUT`
  - `OLLAMA_OVERLOADED`
  - `OLLAMA_CONNECTION_FAILED`

---

### 4) Backend: Reduce payload size before LLM call
Before sending image to Ollama:
- Downscale image to max 512px (long side)
- JPEG encode with quality 60
- Base64 encode compressed bytes only
- Reject payload > 350KB with HTTP 413 and guidance message

---

### 5) Scene-memory-first chat strategy (LLM-on-demand)
Modify `/api/chat` behavior:
- For many questions, answer from `scene_memory` + analytics directly (no LLM call).
- Only call LLM when query requires reasoning/summarization.
- Add a simple router:
  - If query matches count/status/history templates → deterministic response from memory.
  - Else → LLM call with compact context.

This should drastically reduce expensive multimodal inference frequency.

---

### 6) Prompt/context compaction
In `app/llm/prompt_templates.py`:
- Limit context to:
  - last 5–10 events
  - top object counts
  - short scene summary
- Hard token budget for prompt text.
- Never include raw full history buffers.

---

### 7) Observability and error transparency
Add structured logging around chat path:
- request_id
- queue_size
- model_used
- llm_latency_ms
- input_image_kb
- timeout/failure reason

Expose metrics endpoint or include in `/api/health`:
- `llm_queue_size`
- `llm_avg_latency_ms`
- `llm_timeout_count`
- `llm_fallback_count`

---

### 8) Configuration additions (app/config.py)
Add env vars with defaults:

- `CHAT_FRAME_INTERVAL_MS=2000`
- `CHAT_MAX_IMAGE_WIDTH=512`
- `CHAT_JPEG_QUALITY=60`
- `LLM_MAX_QUEUE_SIZE=2`
- `LLM_MAX_CONCURRENT=1`
- `OLLAMA_READ_TIMEOUT=45`
- `OLLAMA_CONNECT_TIMEOUT=5`
- `OLLAMA_MAX_IMAGE_KB=350`
- `LLM_PRIMARY_MODEL=llava:7b`
- `LLM_FALLBACK_MODEL=llava-phi3`
- `LLM_ENABLE_CLOUD_FALLBACK=false`
- `OPENAI_BASE_URL=` (optional)
- `OPENAI_API_KEY=` (optional)

---

### 9) Optional cloud fallback path (feature-flagged)
If `LLM_ENABLE_CLOUD_FALLBACK=true` and local Ollama fails:
- route to cloud-compatible endpoint (OpenAI-style API).
- Keep interface abstraction in `ollama_client.py` (rename to generic `llm_client.py` if needed).
- Must remain OFF by default.

---

### 10) UX safeguards in frontend chat panel
- Show explicit states:
  - “Analyzing frame…”
  - “LLM busy (queued)…”
  - “Using fallback model…”
- On 429, auto-retry with exponential backoff (max 2 retries).
- Disable send button while in-flight for image-based query.

---

## File-Level Implementation Targets
- `frontend/src/components/*Chat*`
- `frontend/src/services/*api*`
- `app/routes/chat_routes.py`
- `app/llm/ollama_client.py` (or rename to `llm_client.py`)
- `app/llm/prompt_templates.py`
- `app/memory/scene_memory.py`
- `app/config.py`
- `app/main.py` (worker startup/shutdown hooks)

## Performance Target After Changes
- Stable single-user operation on i3 + 8GB RAM
- Chat latency: 3–6 seconds typical
- No cascading failures under bursty frame input
- Clear error messages instead of generic `"error"`

## Acceptance Tests
1. Sending rapid chat requests does not crash backend.
2. Queue full returns 429 with structured JSON.
3. Large image payload rejected with 413.
4. Timeout returns `OLLAMA_TIMEOUT` with actionable message.
5. At least 50% fewer LLM calls for count/history-style queries (served from scene memory).
6. `/api/health` reflects queue and latency metrics.