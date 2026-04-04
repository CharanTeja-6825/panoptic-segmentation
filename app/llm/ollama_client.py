"""
Async LLM client with full Vision + Streaming support.

Supports Ollama backend with:
- Structured error types for better observability
- Separate connect/read timeouts
- Fallback model support
- Image compression and validation
- Optional cloud fallback (feature-flagged)
"""

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://100.91.144.84:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_VISION_MODEL = "llava-phi3:latest"


class LLMErrorType(str, Enum):
    """Structured error types for LLM operations."""
    OLLAMA_TIMEOUT = "OLLAMA_TIMEOUT"
    OLLAMA_OVERLOADED = "OLLAMA_OVERLOADED"
    OLLAMA_CONNECTION_FAILED = "OLLAMA_CONNECTION_FAILED"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class LLMError:
    """Structured LLM error with actionable information."""
    error_type: LLMErrorType
    message: str
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    content: str
    model: str
    latency_ms: float
    used_fallback: bool = False
    error: Optional[LLMError] = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass
class LLMMetrics:
    """Metrics for observability."""
    total_requests: int = 0
    successful_requests: int = 0
    timeout_count: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class LLMClient:
    """
    Enhanced LLM client with structured errors, timeouts, and fallback support.

    Optimized for low-resource hardware (8GB RAM, i3 CPU).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        vision_model: str = DEFAULT_VISION_MODEL,
        fallback_model: Optional[str] = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 45.0,
        max_image_kb: int = 350,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.fallback_model = fallback_model
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_image_kb = max_image_kb
        self._available = False
        self.metrics = LLMMetrics()

        # Configure separate connect and read timeouts
        timeout_config = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=30.0,
            pool=5.0,
        )
        self.client = httpx.AsyncClient(timeout=timeout_config)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------

    async def check_health(self) -> bool:
        """Check if Ollama server is available."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/tags",
                timeout=self.connect_timeout,
            )
            self._available = resp.status_code == 200
            return self._available
        except httpx.ConnectError:
            logger.warning("Ollama connection failed: server unreachable")
            self._available = False
            return False
        except httpx.TimeoutException:
            logger.warning("Ollama health check timed out")
            self._available = False
            return False
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    # -------------------------------------------------------------------------
    # List models
    # -------------------------------------------------------------------------

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models on the Ollama server."""
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
            return []

    async def list_vision_models(self) -> List[Dict[str, Any]]:
        """List models suitable for vision tasks (llava-*, etc.)."""
        all_models = await self.list_models()
        vision_keywords = ("llava", "vision", "bakllava", "moondream")
        return [
            m for m in all_models
            if any(kw in m.get("name", "").lower() for kw in vision_keywords)
        ]

    # -------------------------------------------------------------------------
    # Image utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def encode_image_bytes(image_bytes: bytes) -> str:
        """Encode raw image bytes to base64."""
        return base64.b64encode(image_bytes).decode()

    @staticmethod
    def encode_image_file(path: str) -> str:
        """Encode image file to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def validate_image_size(self, image_bytes: bytes) -> Tuple[bool, int]:
        """Validate image size against max_image_kb limit."""
        size_kb = len(image_bytes) // 1024
        return size_kb <= self.max_image_kb, size_kb

    @staticmethod
    def compress_image(
        image_bytes: bytes,
        max_width: int = 512,
        quality: int = 60,
    ) -> bytes:
        """
        Compress image to reduce payload size.

        Args:
            image_bytes: Raw image bytes
            max_width: Maximum width (maintains aspect ratio)
            quality: JPEG quality (1-100)

        Returns:
            Compressed image bytes
        """
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize if needed
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            # Compress to JPEG
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            return output.getvalue()

        except ImportError:
            logger.warning("PIL not available, returning original image")
            return image_bytes
        except Exception as e:
            logger.warning("Image compression failed: %s", e)
            return image_bytes

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    def _classify_error(
        self,
        exception: Exception,
        model: str,
        start_time: float,
    ) -> LLMError:
        """Classify exception into structured error type."""
        latency_ms = (time.time() - start_time) * 1000

        if isinstance(exception, httpx.ConnectError):
            return LLMError(
                error_type=LLMErrorType.OLLAMA_CONNECTION_FAILED,
                message="Cannot connect to Ollama server. Ensure it is running.",
                model=model,
                latency_ms=latency_ms,
            )

        if isinstance(exception, httpx.TimeoutException):
            self.metrics.timeout_count += 1
            return LLMError(
                error_type=LLMErrorType.OLLAMA_TIMEOUT,
                message=f"Request timed out after {self.read_timeout}s. Try a simpler query.",
                model=model,
                latency_ms=latency_ms,
            )

        if isinstance(exception, httpx.HTTPStatusError):
            status = exception.response.status_code
            if status == 503:
                return LLMError(
                    error_type=LLMErrorType.OLLAMA_OVERLOADED,
                    message="Ollama server is overloaded. Try again shortly.",
                    model=model,
                    latency_ms=latency_ms,
                    details={"status_code": status},
                )
            if status == 404:
                return LLMError(
                    error_type=LLMErrorType.MODEL_NOT_FOUND,
                    message=f"Model '{model}' not found. Pull it with: ollama pull {model}",
                    model=model,
                    latency_ms=latency_ms,
                    details={"status_code": status},
                )

        return LLMError(
            error_type=LLMErrorType.UNKNOWN_ERROR,
            message=str(exception),
            model=model,
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------------
    # Chat (text or vision)
    # -------------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        use_fallback: bool = True,
    ) -> LLMResponse:
        """
        Send chat messages to LLM.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (defaults to self.model)
            temperature: Sampling temperature
            use_fallback: Whether to try fallback model on failure

        Returns:
            LLMResponse with content or error
        """
        selected_model = model or self.model
        start_time = time.time()
        self.metrics.total_requests += 1

        payload = {
            "model": selected_model,
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

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.successful_requests += 1
            self.metrics.total_latency_ms += latency_ms

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=selected_model,
                latency_ms=latency_ms,
            )

        except Exception as e:
            error = self._classify_error(e, selected_model, start_time)
            logger.error("Chat error [%s]: %s", error.error_type.value, error.message)

            # Try fallback model if available
            if use_fallback and self.fallback_model and model != self.fallback_model:
                logger.info("Attempting fallback to model: %s", self.fallback_model)
                self.metrics.fallback_count += 1
                fallback_response = await self.chat(
                    messages=messages,
                    model=self.fallback_model,
                    temperature=temperature,
                    use_fallback=False,
                )
                fallback_response.used_fallback = True
                return fallback_response

            return LLMResponse(
                content=f"Error: {error.message}",
                model=selected_model,
                latency_ms=error.latency_ms or 0,
                error=error,
            )

    async def chat_simple(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Simple chat interface returning just the content string."""
        response = await self.chat(messages, model, temperature)
        return response.content

    # -------------------------------------------------------------------------
    # Streaming Chat
    # -------------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Stream chat responses token by token.

        Yields:
            Individual tokens as they arrive
        """
        selected_model = model or self.model
        start_time = time.time()
        self.metrics.total_requests += 1

        payload = {
            "model": selected_model,
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
                            latency_ms = (time.time() - start_time) * 1000
                            self.metrics.successful_requests += 1
                            self.metrics.total_latency_ms += latency_ms
                            return

                    except json.JSONDecodeError:
                        continue

        except httpx.TimeoutException:
            self.metrics.timeout_count += 1
            yield f"\n[Error: Request timed out after {self.read_timeout}s]"

        except httpx.ConnectError:
            yield "\n[Error: Cannot connect to Ollama server]"

        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield f"\n[Error: {e}]"

    # -------------------------------------------------------------------------
    # Generate (completion API)
    # -------------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate completion for a prompt."""
        selected_model = model or self.model
        start_time = time.time()

        payload: Dict[str, Any] = {
            "model": selected_model,
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

            latency_ms = (time.time() - start_time) * 1000
            logger.debug("Generate completed in %.2fms", latency_ms)

            return data.get("response", "")

        except Exception as e:
            error = self._classify_error(e, selected_model, start_time)
            logger.error("Generate error: %s", error.message)
            return f"Error: {error.message}"

    # -------------------------------------------------------------------------
    # Vision helper
    # -------------------------------------------------------------------------

    async def vision_chat(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        model: Optional[str] = None,
        compress: bool = True,
        max_width: int = 512,
        quality: int = 60,
    ) -> LLMResponse:
        """
        Send prompt + images to vision model.

        Args:
            prompt: Text prompt
            images: List of base64 strings or raw bytes
            model: Vision model to use (defaults to self.vision_model)
            compress: Whether to compress images before sending
            max_width: Max image width if compressing
            quality: JPEG quality if compressing

        Returns:
            LLMResponse with content or error
        """
        selected_model = model or self.vision_model
        encoded_images = []

        for img in images:
            if isinstance(img, bytes):
                if compress:
                    img = self.compress_image(img, max_width, quality)

                # Validate size
                valid, size_kb = self.validate_image_size(img)
                if not valid:
                    return LLMResponse(
                        content=f"Error: Image too large ({size_kb}KB > {self.max_image_kb}KB limit)",
                        model=selected_model,
                        latency_ms=0,
                        error=LLMError(
                            error_type=LLMErrorType.IMAGE_TOO_LARGE,
                            message=f"Image size {size_kb}KB exceeds {self.max_image_kb}KB limit",
                            model=selected_model,
                            details={"image_size_kb": size_kb},
                        ),
                    )

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

        return await self.chat(messages, model=selected_model)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for observability."""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "timeout_count": self.metrics.timeout_count,
            "fallback_count": self.metrics.fallback_count,
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
        }


# Backward compatibility alias
OllamaClient = LLMClient
