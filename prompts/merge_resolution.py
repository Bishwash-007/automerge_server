"""Prompts for merge conflict resolution."""

SYSTEM_PROMPT = """You are an expert software engineer specializing in merge conflict resolution.
Your task is to analyze git merge conflicts and provide the best possible resolution.

When resolving conflicts, consider:
1. Code correctness and functionality
2. Project conventions and patterns
3. Which changes preserve more functionality
4. Potential integration issues

You must provide:
1. RESOLVED_CODE: The merged code without any conflict markers
2. SUMMARY: A clear explanation of your resolution decisions

Be concise but thorough in your explanations. Focus on the "why" behind your choices."""

RESOLUTION_PROMPT = """Resolve the following merge conflict.

Language: {language}

Context from codebase (similar patterns and previous resolutions):
{context}

Conflict:
<<<<<<< {head_label}
{head_content}
=======
{incoming_content}
>>>>>>> {incoming_label}

Provide your response in this exact format:

RESOLVED_CODE:
```{language}
[your resolved code here - no conflict markers]
```

SUMMARY:
[Explain which changes you kept and why, any potential issues to review, and your reasoning based on the code patterns]"""

SUMMARY_PROMPT = """Provide a brief summary explaining this merge conflict resolution.

Conflict Context:
- HEAD branch had: {head_summary}
- Incoming branch had: {incoming_summary}

Resolution:
{resolved_code}

Explain in 2-3 sentences:
1. What was the core disagreement between the branches?
2. Why was this resolution chosen?
3. Any follow-up actions the developer should consider?"""

FEW_SHOT_EXAMPLES = [
    {
        "conflict": """<<<<<<< HEAD
def calculate_total(items):
    return sum(item.price for item in items)
=======
def calculate_total(items):
    return sum(item.price * item.quantity for item in items)
>>>>>>> feature-quantity""",
        "resolution": """def calculate_total(items):
    return sum(item.price * item.quantity for item in items)""",
        "summary": "Kept the quantity multiplication from the feature branch as it provides more accurate totals. The HEAD version appears to be a simplified fallback that doesn't account for item quantities.",
    },
    {
        "conflict": """<<<<<<< HEAD
import { useState } from 'react';
=======
import React, { useState } from 'react';
>>>>>>> feature-react-import""",
        "resolution": """import { useState } from 'react';""",
        "summary": "Kept the modern React import style from HEAD. The feature branch uses legacy default import which is unnecessary for React 17+. The current import follows the project's pattern of named imports only.",
    },
]
