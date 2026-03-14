"""
Async Ollama client with full Vision + Streaming support.
Supports frames, images, and text chat.
"""

import base64
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available = False

        # persistent connection (faster)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self.client.aclose()

    # ---------------------------
    # Health check
    # ---------------------------

    async def check_health(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            self._available = resp.status_code == 200
            return self._available
        except Exception:
            self._available = False
            return False

    @property
    def is_available(self):
        return self._available

    # ---------------------------
    # List models
    # ---------------------------

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
            return []

    # ---------------------------
    # Utilities
    # ---------------------------

    @staticmethod
    def encode_image_bytes(image_bytes: bytes) -> str:
        """Encode raw image bytes to base64."""
        return base64.b64encode(image_bytes).decode()

    @staticmethod
    def encode_image_file(path: str) -> str:
        """Encode image file to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # ---------------------------
    # Chat (text or vision)
    # ---------------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            resp = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

            resp.raise_for_status()
            data = resp.json()

            return data.get("message", {}).get("content", "")

        except Exception as e:
            logger.error("Chat error: %s", e)
            return f"Error: {e}"

    # ---------------------------
    # Streaming Chat
    # ---------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as resp:

                resp.raise_for_status()

                async for line in resp.aiter_lines():

                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        token = data.get("message", {}).get("content", "")

                        if token:
                            yield token

                        if data.get("done"):
                            return

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield f"\n[Error: {e}]"

    # ---------------------------
    # Generate (completion API)
    # ---------------------------

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system

        try:
            resp = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )

            resp.raise_for_status()

            data = resp.json()

            return data.get("response", "")

        except Exception as e:
            logger.error("Generate error: %s", e)
            return f"Error: {e}"

    # ---------------------------
    # Vision helper
    # ---------------------------

    async def vision_chat(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        model: Optional[str] = None,
    ) -> str:
        """
        Send prompt + images to vision model.

        images can be:
        - base64 strings
        - raw bytes
        """

        encoded_images = []

        for img in images:
            if isinstance(img, bytes):
                encoded_images.append(self.encode_image_bytes(img))
            else:
                encoded_images.append(img)

        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": encoded_images,
            }
        ]

        return await self.chat(messages, model=model)
