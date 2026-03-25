"""RAG service for orchestrating retrieval-augmented generation."""

import logging
from dataclasses import dataclass

from config import get_settings
from services.conflict_parser import ConflictParser
from services.ollama_client import OllamaClient
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
    retrieved_context: list[dict] | None = None


class RagService:
    """Orchestrates RAG-based conflict resolution."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        retriever: VectorRetriever | None = None,
    ):
        """
        Initialize the RAG service.

        Args:
            ollama_client: Ollama client for generation
            retriever: Vector retriever for context
        """
        self.ollama = ollama_client or OllamaClient()
        self.retriever = retriever or VectorRetriever()

    async def resolve_conflict(
        self,
        conflict_text: str,
        language: str,
        file_path: str | None = None,
        n_context_results: int = 5,
    ) -> ResolutionResult:
        """
        Resolve a merge conflict using RAG.

        Args:
            conflict_text: Full conflict text with markers
            language: Programming language
            file_path: Optional file path for context
            n_context_results: Number of context documents to retrieve

        Returns:
            ResolutionResult with resolved code and summary
        """
        # Parse the conflict
        parsed = ConflictParser.parse(conflict_text)
        if not parsed:
            # Try to extract sections directly
            head_label, head_content, incoming_label, incoming_content = (
                ConflictParser.extract_sections(conflict_text)
            )
        else:
            section = parsed[0].sections[0]
            head_label = section.head_label
            head_content = section.head_content
            incoming_label = section.incoming_label
            incoming_content = section.incoming_content

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
            response = await self.ollama.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Fallback: return HEAD content with low confidence
            return ResolutionResult(
                resolved_code=head_content,
                summary=f"Could not generate resolution: {e}. Showing HEAD content as fallback.",
                confidence=0.1,
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
            retrieved_context=context_docs,
        )

    def resolve_conflict_sync(
        self,
        conflict_text: str,
        language: str,
        file_path: str | None = None,
        n_context_results: int = 5,
    ) -> ResolutionResult:
        """
        Synchronous version of resolve_conflict.

        Args:
            conflict_text: Full conflict text with markers
            language: Programming language
            file_path: Optional file path for context
            n_context_results: Number of context documents to retrieve

        Returns:
            ResolutionResult with resolved code and summary
        """
        # Parse the conflict
        parsed = ConflictParser.parse(conflict_text)
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
                retrieved_context=context_docs,
            )

        # Parse the response
        resolved_code, summary = self._parse_response(response)
        confidence = self._estimate_confidence(context_docs, resolved_code)

        return ResolutionResult(
            resolved_code=resolved_code,
            summary=summary,
            confidence=confidence,
            retrieved_context=context_docs,
        )

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
            # If no clear summary, use the last part of response
            summary = response.split("\n")[-3:]
            summary = "\n".join(summary) if summary else "Resolution generated."

        # Clean up resolved code
        resolved_code = resolved_code.strip()
        if not resolved_code:
            # Fallback: use entire response as code
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

        This is a simple heuristic - can be improved with more signals.
        """
        if not context_docs:
            return 0.5  # Base confidence without context

        # Higher confidence with more relevant context
        avg_relevance = sum(d.get("relevance_score", 0) for d in context_docs) / len(
            context_docs
        )

        # Boost confidence if we have good context
        base_confidence = 0.6 + (avg_relevance * 0.3)

        # Cap at 0.95 (never 100% confident)
        return min(0.95, max(0.1, base_confidence))
