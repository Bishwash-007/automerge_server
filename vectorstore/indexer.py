"""Indexer service for building the RAG knowledge base."""

import hashlib
import logging
import os
import subprocess
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
import yaml

from config.ollama_config import get_settings

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Indexes codebase, git history, and documentation for RAG."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "merge_context",
    ):
        """
        Initialize the vector indexer.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the chroma collection
        """
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name
        self.embedding_model = settings.embedding_model

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
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Created collection '{self.collection_name}'")
        return self._collection

    def index_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> int:
        """
        Index all code files in a directory.

        Args:
            directory: Path to directory to index
            extensions: File extensions to include (e.g., ['.py', '.ts'])
            exclude_patterns: Patterns to exclude (e.g., ['node_modules', '__pycache__'])

        Returns:
            Number of documents indexed
        """
        if extensions is None:
            extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs"]
        if exclude_patterns is None:
            exclude_patterns = [
                "node_modules",
                "__pycache__",
                ".git",
                "venv",
                "dist",
                "build",
            ]

        indexed = 0
        root_path = Path(directory)

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix not in extensions:
                continue

            # Check exclusions
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                if content.strip():
                    self._add_document(
                        content=content,
                        metadata={
                            "type": "source_code",
                            "file_path": str(file_path.relative_to(root_path)),
                            "language": self._detect_language(file_path.suffix),
                        },
                    )
                    indexed += 1
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        logger.info(f"Indexed {indexed} source files")
        return indexed

    def index_git_history(
        self,
        repo_path: str = ".",
        max_commits: int | None = None,
    ) -> int:
        """
        Index git merge commit history.

        Args:
            repo_path: Path to git repository
            max_commits: Maximum number of commits to index

        Returns:
            Number of commits indexed
        """
        settings = get_settings()
        if max_commits is None:
            max_commits = settings.git_history_depth

        indexed = 0

        try:
            # Get merge commits
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    repo_path,
                    "log",
                    "--merges",
                    "-n",
                    str(max_commits),
                    "--pretty=format:%H|%s|%b",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|", 2)
                if len(parts) < 2:
                    continue

                commit_hash = parts[0]
                subject = parts[1]
                body = parts[2] if len(parts) > 2 else ""

                content = f"Merge commit: {subject}\n\n{body}"
                self._add_document(
                    content=content,
                    metadata={
                        "type": "git_history",
                        "commit_hash": commit_hash,
                        "subject": subject,
                    },
                )
                indexed += 1

            logger.info(f"Indexed {indexed} merge commits")

        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
        except Exception as e:
            logger.error(f"Failed to index git history: {e}")

        return indexed

    def index_documentation(
        self,
        docs_dir: str,
        extensions: list[str] | None = None,
    ) -> int:
        """
        Index documentation files.

        Args:
            docs_dir: Path to documentation directory
            extensions: File extensions to include

        Returns:
            Number of documents indexed
        """
        if extensions is None:
            extensions = [".md", ".rst", ".txt"]

        indexed = 0
        docs_path = Path(docs_dir)

        if not docs_path.exists():
            logger.warning(f"Docs directory not found: {docs_path}")
            return 0

        for file_path in docs_path.rglob("*"):
            if not file_path.is_file() or file_path.suffix not in extensions:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                self._add_document(
                    content=content,
                    metadata={
                        "type": "documentation",
                        "file_path": str(file_path.relative_to(docs_path)),
                    },
                )
                indexed += 1
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        logger.info(f"Indexed {indexed} documentation files")
        return indexed

    def index_resolution_examples(
        self,
        examples: list[dict[str, str]],
    ) -> int:
        """
        Index manual resolution examples.

        Args:
            examples: List of dicts with 'conflict', 'resolution', 'explanation' keys

        Returns:
            Number of examples indexed
        """
        indexed = 0

        for example in examples:
            content = f"""
Conflict:
{example.get('conflict', '')}

Resolution:
{example.get('resolution', '')}

Explanation:
{example.get('explanation', '')}
"""
            self._add_document(
                content=content,
                metadata={
                    "type": "resolution_example",
                    "language": example.get("language", "unknown"),
                },
            )
            indexed += 1

        logger.info(f"Indexed {indexed} resolution examples")
        return indexed

    def _add_document(self, content: str, metadata: dict[str, str]) -> None:
        """Add a single document to the index."""
        doc_id = hashlib.sha256(
            f"{content[:100]}:{metadata.get('file_path', '')}".encode()
        ).hexdigest()

        try:
            # Check if document exists
            existing = self.collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                logger.debug(f"Document {doc_id} already exists, skipping")
                return

            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.error(f"Failed to add document: {e}")

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        mapping = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
        }
        return mapping.get(extension, "unknown")

    def clear_index(self) -> None:
        """Clear the entire index."""
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info("Cleared index")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
