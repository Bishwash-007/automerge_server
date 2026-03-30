"""RAG service for orchestrating retrieval-augmented generation."""

import logging
from dataclasses import dataclass
from typing import Literal

from config.ollama_config import get_settings
from config.hf_config import get_settings as get_hf_settings
from services.conflict_parser import ConflictParser
from services.ollama_client import OllamaClient
from services.hf_client import HFClient
from prompts.merge_resolution import (
    SYSTEM_PROMPT,
    RESOLUTION_PROMPT,
    SUMMARY_PROMPT,
)
from vectorstore.retriever import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of a conflict resolution."""

    resolved_code: str
    summary: str
    confidence: float
    provider: str = "ollama"
    retrieved_context: list[dict] | None = None


class RagService:
    """Orchestrates RAG-based conflict resolution."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        hf_client: HFClient | None = None,
        retriever: VectorRetriever | None = None,
        default_provider: Literal["ollama", "huggingface"] = "ollama",
    ):
        """
        Initialize the RAG service.

        Args:
            ollama_client: Ollama client for generation
            hf_client: HuggingFace client for generation
            retriever: Vector retriever for context
            default_provider: Default LLM provider to use
        """
        self.ollama = ollama_client or OllamaClient()
        self.hf = hf_client or HFClient()
        self.retriever = retriever or VectorRetriever()
        self.default_provider = default_provider

    def _get_client(self, provider: Literal["ollama", "huggingface"]) -> OllamaClient | HFClient:
        """Get the appropriate client based on provider."""
        return self.ollama if provider == "ollama" else self.hf

    async def resolve_conflict(
        self,
        conflict_text: str,
        language: str,
        file_path: str | None = None,
        n_context_results: int = 5,
        provider: Literal["ollama", "huggingface"] | None = None,
    ) -> ResolutionResult:
        """
        Resolve a merge conflict using RAG.

        Args:
            conflict_text: Full conflict text with markers
            language: Programming language
            file_path: Optional file path for context
            n_context_results: Number of context documents to retrieve
            provider: LLM provider to use (uses default if not specified)

        Returns:
            ResolutionResult with resolved code and summary
        """
        provider = provider or self.default_provider

        # Parse the conflict
        parsed = ConflictParser.parse(conflict_text)
        base_content = None
        if not parsed:
            head_label, head_content, incoming_label, incoming_content = (
                ConflictParser.extract_sections(conflict_text)
            )
        else:
            section = parsed[0].sections[0]
            head_label = section.head_label
            head_content = section.head_content
            incoming_label = section.incoming_label
            incoming_content = section.incoming_content
            base_content = section.base_content

        # For HuggingFace, use the specialized resolve_conflict method
        if provider == "huggingface":
            base = base_content if base_content else head_content
            return await self._resolve_with_hf(
                base=base,
                ours=head_content,
                theirs=incoming_content,
                language=language,
                provider=provider,
            )

        # Retrieve relevant context for Ollama
        context_docs = self.retriever.retrieve_for_conflict(
            head_content=head_content,
            incoming_content=incoming_content,
            language=language,
            file_path=file_path,
            n_results=n_context_results,
        )

        # Format context for prompt
        context_str = self._format_context(context_docs)

        # Build the resolution prompt
        user_prompt = RESOLUTION_PROMPT.format(
            language=language,
            head_label=head_label,
            head_content=head_content,
            incoming_label=incoming_label,
            incoming_content=incoming_content,
            context=context_str,
        )

        # Generate resolution using Ollama
        try:
            response = await self.ollama.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ResolutionResult(
                resolved_code=head_content,
                summary=f"Could not generate resolution: {e}. Showing HEAD content as fallback.",
                confidence=0.1,
                provider=provider,
                retrieved_context=context_docs,
            )

        # Parse the response to extract resolution and summary
        resolved_code, summary = self._parse_response(response)

        # Estimate confidence based on context relevance
        confidence = self._estimate_confidence(context_docs, resolved_code)

        return ResolutionResult(
            resolved_code=resolved_code,
            summary=summary,
            confidence=confidence,
            provider=provider,
            retrieved_context=context_docs,
        )

    async def _resolve_with_hf(
        self,
        base: str,
        ours: str,
        theirs: str,
        language: str,
        provider: str,
    ) -> ResolutionResult:
        """Resolve conflict using HuggingFace CodeT5 model, then generate summary with Ollama."""
        try:
            resolved_code = self.hf.resolve_conflict(
                base=base,
                ours=ours,
                theirs=theirs,
                language=language,
            )

            # Generate a detailed summary using Ollama
            summary = await self._generate_summary(
                head_content=ours,
                incoming_content=theirs,
                resolved_code=resolved_code,
                language=language,
            )

            return ResolutionResult(
                resolved_code=resolved_code,
                summary=summary,
                confidence=0.75,
                provider=provider,
            )
        except Exception as e:
            logger.error(f"HF resolution failed: {e}")
            return ResolutionResult(
                resolved_code=ours,
                summary=f"Could not generate resolution: {e}. Showing HEAD content as fallback.",
                confidence=0.1,
                provider=provider,
            )

    async def _generate_summary(
        self,
        head_content: str,
        incoming_content: str,
        resolved_code: str,
        language: str,
    ) -> str:
        """Generate a detailed summary for an HF resolution using Ollama."""
        try:
            user_prompt = SUMMARY_PROMPT.format(
                head_summary=self._summarize_code(head_content, language),
                incoming_summary=self._summarize_code(incoming_content, language),
                resolved_code=resolved_code,
            )
            response = await self.ollama.chat(
                system_prompt="You are an expert software engineer explaining merge conflict resolutions.",
                user_prompt=user_prompt,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return "Resolved using HuggingFace CodeT5 model. Summary unavailable."

    def _summarize_code(self, code: str, language: str) -> str:
        """Create a brief summary of code changes."""
        if not code.strip():
            return "empty"
        lines = code.strip().split('\n')
        if len(lines) <= 2:
            return code.strip()
        first = lines[0].strip()
        last = lines[-1].strip()
        return f"{first} ... {last} ({len(lines)} lines)"

    def resolve_conflict_sync(
        self,
        conflict_text: str,
        language: str,
        file_path: str | None = None,
        n_context_results: int = 5,
        provider: Literal["ollama", "huggingface"] | None = None,
    ) -> ResolutionResult:
        """
        Synchronous version of resolve_conflict.

        Args:
            conflict_text: Full conflict text with markers
            language: Programming language
            file_path: Optional file path for context
            n_context_results: Number of context documents to retrieve
            provider: LLM provider to use

        Returns:
            ResolutionResult with resolved code and summary
        """
        provider = provider or self.default_provider

        # Parse the conflict
        parsed = ConflictParser.parse(conflict_text)
        base_content = None
        if not parsed:
            head_label, head_content, incoming_label, incoming_content = (
                ConflictParser.extract_sections(conflict_text)
            )
        else:
            section = parsed[0].sections[0]
            head_label = section.head_label
            head_content = section.head_content
            incoming_label = section.incoming_label
            incoming_content = section.incoming_content
            base_content = section.base_content

        # For HuggingFace, use the specialized resolve_conflict method
        if provider == "huggingface":
            base = base_content if base_content else head_content
            return self._resolve_with_hf_sync(
                base=base,
                ours=head_content,
                theirs=incoming_content,
                language=language,
                provider=provider,
            )

        # Retrieve relevant context
        context_docs = self.retriever.retrieve_for_conflict(
            head_content=head_content,
            incoming_content=incoming_content,
            language=language,
            file_path=file_path,
            n_results=n_context_results,
        )

        # Format context for prompt
        context_str = self._format_context(context_docs)

        # Build the resolution prompt
        user_prompt = RESOLUTION_PROMPT.format(
            language=language,
            head_label=head_label,
            head_content=head_content,
            incoming_label=incoming_label,
            incoming_content=incoming_content,
            context=context_str,
        )

        # Generate resolution
        try:
            response = self.ollama.chat_sync(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ResolutionResult(
                resolved_code=head_content,
                summary=f"Could not generate resolution: {e}. Showing HEAD content as fallback.",
                confidence=0.1,
                provider=provider,
                retrieved_context=context_docs,
            )

        # Parse the response
        resolved_code, summary = self._parse_response(response)
        confidence = self._estimate_confidence(context_docs, resolved_code)

        return ResolutionResult(
            resolved_code=resolved_code,
            summary=summary,
            confidence=confidence,
            provider=provider,
            retrieved_context=context_docs,
        )

    def _resolve_with_hf_sync(
        self,
        base: str,
        ours: str,
        theirs: str,
        language: str,
        provider: str,
    ) -> ResolutionResult:
        """Synchronous resolve using HuggingFace CodeT5 model, then generate summary with Ollama."""
        try:
            resolved_code = self.hf.resolve_conflict(
                base=base,
                ours=ours,
                theirs=theirs,
                language=language,
            )

            # Generate a detailed summary using Ollama
            summary = self._generate_summary_sync(
                head_content=ours,
                incoming_content=theirs,
                resolved_code=resolved_code,
                language=language,
            )

            return ResolutionResult(
                resolved_code=resolved_code,
                summary=summary,
                confidence=0.75,
                provider=provider,
            )
        except Exception as e:
            logger.error(f"HF resolution failed: {e}")
            return ResolutionResult(
                resolved_code=ours,
                summary=f"Could not generate resolution: {e}. Showing HEAD content as fallback.",
                confidence=0.1,
                provider=provider,
            )

    def _generate_summary_sync(
        self,
        head_content: str,
        incoming_content: str,
        resolved_code: str,
        language: str,
    ) -> str:
        """Generate a detailed summary for an HF resolution using Ollama (sync version)."""
        try:
            user_prompt = SUMMARY_PROMPT.format(
                head_summary=self._summarize_code(head_content, language),
                incoming_summary=self._summarize_code(incoming_content, language),
                resolved_code=resolved_code,
            )
            response = self.ollama.chat_sync(
                system_prompt="You are an expert software engineer explaining merge conflict resolutions.",
                user_prompt=user_prompt,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return "Resolved using HuggingFace CodeT5 model. Summary unavailable."

    def _format_context(self, context_docs: list[dict]) -> str:
        """Format retrieved context documents for the prompt."""
        if not context_docs:
            return "No relevant context found in the codebase."

        formatted = []
        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get("metadata", {})
            file_path = metadata.get("file_path", "unknown")
            doc_type = metadata.get("type", "unknown")
            relevance = doc.get("relevance_score", 0)

            formatted.append(
                f"[Example {i}] (Type: {doc_type}, File: {file_path}, Relevance: {relevance:.2f})\n"
                f"```\n{doc.get('content', '')}\n```"
            )

        return "\n\n".join(formatted)

    def _parse_response(self, response: str) -> tuple[str, str]:
        """
        Parse LLM response to extract resolution and summary.

        Expected format:
        RESOLVED_CODE:
        ```language
        ... code ...
        ```

        SUMMARY:
        ... explanation ...
        """
        resolved_code = ""
        summary = ""

        # Try to extract code block
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 2:
                resolved_code = parts[1].strip()
                # Remove language identifier if present
                if "\n" in resolved_code:
                    lines = resolved_code.split("\n")
                    if lines[0] and not lines[0].startswith((" ", "\t")):
                        resolved_code = "\n".join(lines[1:])

        # Try to extract summary
        if "SUMMARY:" in response:
            summary_parts = response.split("SUMMARY:", 1)
            summary = summary_parts[1].strip()
        elif "Explanation:" in response:
            summary_parts = response.split("Explanation:", 1)
            summary = summary_parts[1].strip()
        else:
            summary = response.split("\n")[-3:]
            summary = "\n".join(summary) if summary else "Resolution generated."

        # Clean up resolved code
        resolved_code = resolved_code.strip()
        if not resolved_code:
            resolved_code = response.split("SUMMARY:")[0].strip()
            resolved_code = resolved_code.split("Explanation:")[0].strip()

        return resolved_code, summary

    def _estimate_confidence(
        self,
        context_docs: list[dict],
        resolved_code: str,
    ) -> float:
        """
        Estimate confidence score based on context relevance.
        """
        if not context_docs:
            return 0.5

        avg_relevance = sum(d.get("relevance_score", 0) for d in context_docs) / len(
            context_docs
        )

        base_confidence = 0.6 + (avg_relevance * 0.3)
        return min(0.95, max(0.1, base_confidence))
