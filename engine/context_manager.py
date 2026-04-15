"""
Context Manager — intelligent context window management.

Inspired by Claude Code's "just-in-time" context loading.
Instead of dumping everything into the prompt, we:
1. Always include: system prompt + current task
2. Conditionally include: relevant memory entries
3. On-demand: tool results (trimmed if needed)
4. Auto-trim oldest context when approaching budget
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("gemma-harness.context")


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate. Gemma uses ~1.3 tokens per word for English,
    ~2-3 tokens per character for CJK. Use a conservative heuristic.
    """
    # Count CJK characters
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    non_cjk = len(text) - cjk_count
    return int(cjk_count * 2.5 + non_cjk / 3.5)


class ContextManager:
    """Manages the conversation context within a token budget."""

    def __init__(self, budget: int = config.MAX_CONTEXT_TOKENS):
        self.budget = budget
        self._system_prompt: str = ""
        self._system_tokens: int = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt (always included, never trimmed)."""
        self._system_prompt = prompt
        self._system_tokens = _estimate_tokens(prompt)
        logger.info("System prompt set: ~%d tokens", self._system_tokens)

    def build_messages(
        self,
        conversation: List[Dict[str, Any]],
        memory_keys: Optional[List[str]] = None,
        execution_brief: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Build the final message list within budget.

        Priority (never trimmed → first to trim):
        1. System prompt (never trimmed)
        2. Last user message (never trimmed)
        3. Last assistant + tool results (high priority)
        4. Memory entries (medium priority)
        5. Earlier conversation turns (low priority, trimmed first)
        """
        messages: list[dict[str, Any]] = []
        used_tokens = 0

        # 1. System prompt
        system_content = self._system_prompt
        if memory_keys:
            memory_block = self._load_memories(memory_keys)
            if memory_block:
                system_content += f"\n\n## Recalled Memories\n{memory_block}"
        if execution_brief:
            system_content += f"\n\n## Execution Brief\n{execution_brief}"

        messages.append({"role": "system", "content": system_content})
        used_tokens += _estimate_tokens(system_content)

        if not conversation:
            return messages

        # 2. Reserve space for the last user message (never trim)
        last_msg = conversation[-1]
        last_tokens = _estimate_tokens(str(last_msg.get("content", "")))
        reserved = used_tokens + last_tokens

        # 3. Fit as many earlier messages as possible within budget
        remaining_budget = self.budget - reserved
        middle_messages: list[dict[str, Any]] = []

        # Process conversation in reverse (most recent = highest priority)
        for msg in reversed(conversation[:-1]):
            msg_tokens = _estimate_tokens(str(msg.get("content", "")))

            # Tool calls / tool results are usually compact, always include
            if msg.get("role") == "tool" or msg.get("tool_calls"):
                msg_tokens = min(msg_tokens, 1500)  # cap tool content

            if msg_tokens <= remaining_budget:
                middle_messages.insert(0, msg)
                remaining_budget -= msg_tokens
            else:
                # Try to include a trimmed version
                trimmed = self._trim_message(msg, remaining_budget)
                if trimmed:
                    middle_messages.insert(0, trimmed)
                    remaining_budget -= _estimate_tokens(str(trimmed.get("content", "")))
                break  # Can't fit more, stop

        messages.extend(middle_messages)
        messages.append(last_msg)

        total_tokens = self.budget - remaining_budget
        logger.debug(
            "Context built: %d messages, ~%d/%d tokens",
            len(messages), total_tokens, self.budget,
        )
        return messages

    @staticmethod
    def _load_memories(keys: List[str]) -> str:
        """Load memory entries by key."""
        parts = []
        for key in keys[:5]:  # Max 5 memories to keep context reasonable
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            path = config.MEMORY_DIR / f"{safe_key}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    parts.append(f"**{key}**: {data.get('content', '')[:500]}")
                except (json.JSONDecodeError, OSError):
                    continue
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _trim_message(msg: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """Trim a message to fit within a token budget."""
        content = str(msg.get("content", ""))
        if not content:
            return None

        # Rough trim: keep first ~max_tokens worth of characters
        char_budget = int(max_tokens * 3)  # approximate chars per token
        if len(content) > char_budget:
            trimmed_content = content[:char_budget] + "\n[... trimmed ...]"
            return {**msg, "content": trimmed_content}
        return msg

    @staticmethod
    def auto_detect_memory_keys(task: str) -> list[str]:
        """
        Scan the memory directory and suggest relevant keys
        based on simple keyword matching against the task.
        """
        if not config.MEMORY_DIR.exists():
            return []

        task_lower = task.lower()
        relevant = []
        for f in config.MEMORY_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                key = data.get("key", f.stem)
                content = data.get("content", "")
                # Simple keyword overlap check
                key_words = set(key.lower().replace("_", " ").replace("-", " ").split())
                task_words = set(task_lower.replace("_", " ").replace("-", " ").split())
                if key_words & task_words:
                    relevant.append(key)
            except (json.JSONDecodeError, OSError):
                continue

        return relevant[:3]  # Max 3 auto-loaded memories
