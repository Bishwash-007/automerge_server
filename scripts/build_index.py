#!/usr/bin/env python3
"""Script to build the RAG index from git history and codebase."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.indexer import VectorIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build RAG index for merge conflict resolution")
    parser.add_argument(
        "--codebase",
        type=str,
        default=".",
        help="Path to codebase directory to index",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=None,
        help="Path to documentation directory",
    )
    parser.add_argument(
        "--git-history",
        action="store_true",
        help="Index git merge commit history",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before building",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=500,
        help="Maximum number of git commits to index",
    )

    args = parser.parse_args()

    logger.info("Initializing VectorIndexer...")
    indexer = VectorIndexer()

    if args.clear:
        logger.info("Clearing existing index...")
        indexer.clear_index()

    total_indexed = 0

    # Index codebase
    if args.codebase:
        codebase_path = Path(args.codebase)
        if codebase_path.exists():
            logger.info(f"Indexing codebase at {codebase_path}...")
            count = indexer.index_directory(
                str(codebase_path),
                exclude_patterns=[
                    "node_modules",
                    "__pycache__",
                    ".git",
                    "venv",
                    "dist",
                    "build",
                    "automerge_server/chroma_db",
                ],
            )
            total_indexed += count
            logger.info(f"Indexed {count} code files")
        else:
            logger.warning(f"Codebase path not found: {codebase_path}")

    # Index git history
    if args.git_history:
        logger.info("Indexing git merge history...")
        count = indexer.index_git_history(
            repo_path=args.codebase,
            max_commits=args.max_commits,
        )
        total_indexed += count
        logger.info(f"Indexed {count} merge commits")

    # Index documentation
    if args.docs:
        docs_path = Path(args.docs)
        if docs_path.exists():
            logger.info(f"Indexing documentation at {docs_path}...")
            count = indexer.index_documentation(str(docs_path))
            total_indexed += count
            logger.info(f"Indexed {count} documentation files")
        else:
            logger.warning(f"Documentation path not found: {docs_path}")

    logger.info(f"Indexing complete. Total documents indexed: {total_indexed}")


if __name__ == "__main__":
    main()
