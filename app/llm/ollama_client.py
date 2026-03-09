"""
Ollama LLM client module.

Provides an async HTTP client for the Ollama REST API with streaming
support, prompt template management, and scene-to-text encoding.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default Ollama configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaClient:
    """Async client for the Ollama REST API.

    Args:
        base_url: Ollama server URL.
        model: Default model name.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available = False

    async def check_health(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                self._available = resp.status_code == 200
                return self._available
        except Exception:
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on the Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("models", [])
        except Exception as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
        return []

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion request (non-streaming).

        Args:
            messages: List of {role, content} message dicts.
            model: Model override.
            temperature: Sampling temperature.

        Returns:
            The assistant's response text.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except Exception as exc:
            logger.error("Ollama chat error: %s", exc)
            return f"Error communicating with LLM: {exc}"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response token by token.

        Yields:
            Text chunks as they arrive from Ollama.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            content = (
                                chunk.get("message", {}).get("content", "")
                            )
                            if content:
                                yield content
                            if chunk.get("done", False):
                                return
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.error("Ollama stream error: %s", exc)
            yield f"\n[Error: {exc}]"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str:
        """Send a generate (completion) request.

        Args:
            prompt: The text prompt.
            model: Model override.
            system: Optional system prompt.

        Returns:
            Generated text.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except Exception as exc:
            logger.error("Ollama generate error: %s", exc)
            return f"Error: {exc}"
