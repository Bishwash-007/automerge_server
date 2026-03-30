"""Retriever service for fetching relevant context from vector store."""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.ollama_config import get_settings

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieves relevant context from the vector store."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "merge_context",
    ):
        """
        Initialize the vector retriever.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the chroma collection
        """
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name

        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
            except Exception:
                logger.warning(
                    f"Collection '{self.collection_name}' not found. "
                    "Run the indexer first."
                )
                self._collection = None
        return self._collection

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        language: str | None = None,
        file_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query text
            n_results: Number of results to return
            language: Optional language filter
            file_path: Optional file path for context

        Returns:
            List of relevant documents with metadata
        """
        if self.collection is None:
            logger.debug("No collection available, returning empty results")
            return []

        try:
            # Build where clause for filtering
            where_clause = None
            if language:
                where_clause = {"language": language}

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            # Format results
            formatted = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = (
                    results["distances"][0][i] if results["distances"] else None
                )
                formatted.append(
                    {
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "relevance_score": 1 - (distance or 1),
                    }
                )

            return formatted

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def retrieve_for_conflict(
        self,
        head_content: str,
        incoming_content: str,
        language: str,
        file_path: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve context specifically for merge conflict resolution.

        Args:
            head_content: Content from HEAD branch
            incoming_content: Content from incoming branch
            language: Programming language
            file_path: Optional file path
            n_results: Number of results

        Returns:
            List of relevant examples for conflict resolution
        """
        # Build query from both versions
        query = f"{language} code: {head_content}\n{incoming_content}"
        if file_path:
            query = f"File: {file_path}\n{query}"

        return self.retrieve(query, n_results=n_results, language=language)

    def get_similar_resolutions(
        self,
        conflict_pattern: str,
        language: str,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find similar past resolutions.

        Args:
            conflict_pattern: Pattern or description of the conflict
            language: Programming language
            n_results: Number of results

        Returns:
            List of similar resolution examples
        """
        return self.retrieve(
            f"merge conflict resolution {language}: {conflict_pattern}",
            n_results=n_results,
            language=language,
        )
