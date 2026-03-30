"""Service for parsing git conflict markers."""

import re
from dataclasses import dataclass


@dataclass
class ConflictSection:
    """Represents a section of a merge conflict."""

    head_content: str
    incoming_content: str
    base_content: str | None = None
    head_label: str = "HEAD"
    incoming_label: str = "incoming"


@dataclass
class ParsedConflict:
    """Represents a parsed merge conflict with extracted sections."""

    original_text: str
    sections: list[ConflictSection]
    has_multiple_conflicts: bool = False


class ConflictParser:
    """Parser for git merge conflict markers."""

    # Pattern to match conflict markers (two-way: HEAD vs incoming)
    CONFLICT_PATTERN = re.compile(
        r"<<<<<<<\s*(\w+)?\s*\n"  # Start with HEAD label
        r"([\s\S]*?)"  # HEAD content
        r"\n=======\n"  # Separator
        r"([\s\S]*?)"  # Incoming content
        r"\n>>>>>>>[ ]*(\w+)?",  # End with incoming label
        re.MULTILINE,
    )

    # Pattern for three-way conflicts with base section
    THREE_WAY_PATTERN = re.compile(
        r"<<<<<<<\s*(\w+)?\s*\n"
        r"([\s\S]*?)"
        r"\n=======\n"
        r"([\s\S]*?)"
        r"\n\|\|\|\|\|\|\|\s*(\w+)?\s*\n"
        r"([\s\S]*?)"
        r"\n=======\n"
        r"([\s\S]*?)"
        r"\n>>>>>>>[ ]*(\w+)?",
        re.MULTILINE,
    )

    @classmethod
    def parse(cls, conflict_text: str) -> list[ParsedConflict]:
        """
        Parse conflict text and extract all conflicts.

        Args:
            conflict_text: Text containing one or more merge conflicts

        Returns:
            List of ParsedConflict objects
        """
        conflicts = []

        # First try three-way conflict pattern
        three_way_matches = list(cls.THREE_WAY_PATTERN.finditer(conflict_text))

        for match in three_way_matches:
            head_label = match.group(1) or "HEAD"
            head_content = match.group(2).strip()
            incoming_content = match.group(3).strip()
            base_label = match.group(4) or "base"
            base_content = match.group(5).strip()
            incoming_content_merged = match.group(6).strip()
            incoming_label = match.group(7) or "incoming"

            # Use the incoming_content from the second section (after base)
            section = ConflictSection(
                head_content=head_content,
                incoming_content=incoming_content_merged,
                base_content=base_content,
                head_label=head_label,
                incoming_label=incoming_label,
            )

            parsed = ParsedConflict(
                original_text=match.group(0),
                sections=[section],
                has_multiple_conflicts=len(three_way_matches) > 1,
            )
            conflicts.append(parsed)

        # Fall back to two-way conflict pattern (if no three-way found)
        if not three_way_matches:
            two_way_matches = list(cls.CONFLICT_PATTERN.finditer(conflict_text))

            for match in two_way_matches:
                head_label = match.group(1) or "HEAD"
                head_content = match.group(2).strip()
                incoming_content = match.group(3).strip()
                incoming_label = match.group(4) or "incoming"

                section = ConflictSection(
                    head_content=head_content,
                    incoming_content=incoming_content,
                    base_content=None,
                    head_label=head_label,
                    incoming_label=incoming_label,
                )

                parsed = ParsedConflict(
                    original_text=match.group(0),
                    sections=[section],
                    has_multiple_conflicts=len(two_way_matches) > 1,
                )
                conflicts.append(parsed)

        return conflicts

    @classmethod
    def extract_sections(cls, conflict_text: str) -> tuple[str, str, str, str]:
        """
        Extract conflict sections from a single conflict.

        Args:
            conflict_text: Text containing a single merge conflict

        Returns:
            Tuple of (head_label, head_content, incoming_label, incoming_content)
        """
        match = cls.CONFLICT_PATTERN.search(conflict_text)
        if not match:
            return ("HEAD", "", "incoming", "")

        head_label = match.group(1) or "HEAD"
        head_content = match.group(2).strip()
        incoming_content = match.group(3).strip()
        incoming_label = match.group(4) or "incoming"

        return (head_label, head_content, incoming_label, incoming_content)

    @classmethod
    def is_valid_conflict(cls, text: str) -> bool:
        """Check if text contains valid conflict markers."""
        return bool(cls.CONFLICT_PATTERN.search(text))

    @classmethod
    def count_conflicts(cls, text: str) -> int:
        """Count the number of conflicts in text."""
        return len(cls.CONFLICT_PATTERN.findall(text))
