"""Unified LLM client factory for choosing between Ollama and HuggingFace."""

import logging
from typing import Literal

from config.ollama_config import get_settings as get_ollama_settings
from config.hf_config import get_settings as get_hf_settings
from services.ollama_client import OllamaClient
from services.hf_client import HFClient

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client that can use either Ollama or HuggingFace based on provider selection."""

    def __init__(
        self,
        provider: Literal["ollama", "huggingface"] = "ollama",
        **kwargs,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: Which LLM provider to use - "ollama" or "huggingface"
            **kwargs: Additional arguments passed to the underlying client
        """
        self.provider = provider

        if provider == "ollama":
            settings = get_ollama_settings()
            self._client = OllamaClient(
                base_url=kwargs.get("base_url", settings.ollama_base_url),
                model=kwargs.get("model", settings.ollama_model),
            )
        elif provider == "huggingface":
            settings = get_hf_settings()
            self._client = HFClient(
                model_name=kwargs.get("model", settings.hf_model),
            )
            # HF-specific settings
            self._client.max_length = kwargs.get("max_length", settings.hf_max_length)
            self._client.num_beams = kwargs.get("num_beams", settings.hf_num_beams)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'huggingface'.")

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        **kwargs,
    ):
        """
        Send a chat request (Ollama only).

        Args:
            system_prompt: System instruction prompt
            user_prompt: User message prompt
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            Generated response text
        """
        if self.provider == "ollama":
            return await self._client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "chat() is only available for Ollama provider. "
                "Use resolve_conflict() for HuggingFace."
            )

    def chat_sync(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, **kwargs):
        """Synchronous chat (Ollama only)."""
        if self.provider == "ollama":
            return self._client.chat_sync(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "chat_sync() is only available for Ollama provider. "
                "Use resolve_conflict() for HuggingFace."
            )

    def resolve_conflict(self, base: str, ours: str, theirs: str, language: str = "python") -> str:
        """
        Resolve a merge conflict using the HuggingFace CodeT5 model.

        Args:
            base: Base version code
            ours: Our version code
            theirs: Their version code
            language: Programming language

        Returns:
            Resolved code as a string
        """
        if self.provider == "huggingface":
            return self._client.resolve_conflict(
                base=base,
                ours=ours,
                theirs=theirs,
                language=language,
            )
        else:
            raise NotImplementedError(
                "resolve_conflict() is only available for HuggingFace provider. "
                "Use chat() for Ollama."
            )

    async def health_check(self) -> dict:
        """Check health of the underlying client."""
        if self.provider == "ollama":
            return await self._client.health_check()
        else:
            # HuggingFace models are local, just check if model loaded
            return {
                "provider": "huggingface",
                "model": self._client.model.name_or_path
                if hasattr(self._client.model, "name_or_path")
                else str(self._client.model),
                "status": "loaded",
            }


def get_llm_client(provider: Literal["ollama", "huggingface"] = "ollama") -> LLMClient:
    """Factory function to get an LLM client."""
    return LLMClient(provider=provider)
