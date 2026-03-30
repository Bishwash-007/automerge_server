"""Ollama client service for interacting with Qwen 3.5."""

import asyncio
import logging
from typing import Any

import ollama
from ollama import ChatResponse

from config.ollama_config import get_settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the Ollama client.

        Args:
            base_url: Ollama server URL (default from config)
            model: Model name to use (default from config)
        """
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self._client = ollama.Client(host=self.base_url)

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat request to Ollama.

        Args:
            system_prompt: System instruction prompt
            user_prompt: User message prompt
            temperature: Sampling temperature (lower = more deterministic)
            **kwargs: Additional options for the chat request

        Returns:
            Generated response text
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={
                        "temperature": temperature,
                        "top_p": 0.9,
                        **kwargs,
                    },
                ),
            )
            return self._extract_content(response)
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    def chat_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous version of chat.

        Args:
            system_prompt: System instruction prompt
            user_prompt: User message prompt
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            Generated response text
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    **kwargs,
                },
            )
            return self._extract_content(response)
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    def _extract_content(self, response: ChatResponse) -> str:
        """Extract content from chat response."""
        if hasattr(response, "message") and response.message:
            return response.message.get("content", "")
        return ""

    async def health_check(self) -> dict[str, Any]:
        """
        Check if Ollama server is available and model is loaded.

        Returns:
            Health status dictionary
        """
        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self._client.list)

            model_names = []
            if hasattr(models, "models"):
                model_names = [m.model for m in models.models]

            model_available = self.model in model_names

            return {
                "ollama_available": True,
                "model_loaded": self.model if model_available else None,
                "available_models": model_names,
            }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "ollama_available": False,
                "model_loaded": None,
                "error": str(e),
            }

    def is_model_available(self) -> bool:
        """Check if the configured model is available."""
        try:
            models = self._client.list()
            model_names = [m.model for m in models.models] if hasattr(models, "models") else []
            return self.model in model_names
        except Exception:
            return False
