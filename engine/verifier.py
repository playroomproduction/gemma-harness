"""
Response Verifier — multi-pass quality gate.

Catches common Gemma 4 failure modes before the response
reaches the user. If verification fails, the agent loop
can re-prompt with feedback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifyResult:
    """Result of a verification pass."""
    passed: bool
    reasons: list[str] = field(default_factory=list)


class ResponseVerifier:
    """
    Post-generation quality gate.

    Runs a series of checks against the model's response
    and the original task to catch common issues.
    """

    def verify(
        self,
        response: str,
        task: str,
        messages: list[dict[str, Any]],
    ) -> VerifyResult:
        """
        Run all verification checks.
        Returns VerifyResult with pass/fail and reasons.
        """
        reasons: list[str] = []

        # 1. Not empty
        if not response.strip():
            reasons.append("回覆為空")
            return VerifyResult(passed=False, reasons=reasons)

        # 2. No tool hesitation in final response
        if self._check_tool_hesitation(response):
            reasons.append("回覆中描述咗會用工具但冇實際用，請直接執行工具或者直接回答")

        # 3. No excessive repetition
        if self._check_repetition(response):
            reasons.append("回覆包含重複段落，請去除重複內容")

        # 4. Completeness check (if task has multiple parts)
        missing = self._check_completeness(response, task)
        if missing:
            reasons.append(f"回覆可能未完整回答：{missing}")

        # 5. Language consistency
        if self._check_language_drift(response, task):
            reasons.append("回覆語言同 task 唔一致（用戶用中文但回覆用英文，或反之）")

        # 6. No raw error traces
        if self._check_error_leak(response):
            reasons.append("回覆包含 raw error traceback，請用人話解釋錯誤")

        return VerifyResult(
            passed=len(reasons) == 0,
            reasons=reasons,
        )

    @staticmethod
    def _check_tool_hesitation(text: str) -> bool:
        """Detect model describing tool use instead of doing it."""
        patterns = [
            r"我(會|可以|需要)(先)?用.*工具",
            r"(需要|要)(先)?調用",
            r"讓我(先)?(調用|使用|查詢)",
            r"是否需要我.*工具",
            r"要唔要我.*查",
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _check_repetition(text: str) -> bool:
        """Detect repeated paragraphs or sentences."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            return False

        seen: set[str] = set()
        duplicates = 0
        for para in paragraphs:
            # Normalize for comparison
            normalized = re.sub(r"\s+", " ", para).strip().lower()
            if len(normalized) > 30 and normalized in seen:
                duplicates += 1
            seen.add(normalized)

        return duplicates >= 1

    @staticmethod
    def _check_completeness(response: str, task: str) -> str:
        """
        Check if numbered/bulleted items in the task are all addressed.
        Returns a description of missing items, or empty string if complete.
        """
        # Find numbered items in task
        numbered = re.findall(r"(?:^|\n)\s*(\d+)[.、)\]]\s*(.+)", task)
        if len(numbered) < 2:
            return ""  # Not a multi-part task

        # Simple check: are there at least as many distinct sections in response?
        response_sections = len(re.findall(r"(?:^|\n)\s*(?:#{1,3}\s|[-*]\s|\d+[.、)]\s)", response))
        if response_sections < len(numbered):
            return f"Task 有 {len(numbered)} 個要求，但回覆似乎只涵蓋 {response_sections} 項"

        return ""

    @staticmethod
    def _check_language_drift(response: str, task: str) -> bool:
        """Check if response language mismatches task language."""
        import re
        # Strip code blocks and technical content before checking
        cleaned = re.sub(r'```.*?```', '', response, flags=re.S)
        cleaned = re.sub(r'`[^`]+`', '', cleaned)
        # Strip lines that look like reasoning (UNDERSTAND, PLAN, etc.)
        cleaned = re.sub(r'^\*\*[A-Z]+\*\*:.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        if not cleaned:
            return False

        task_cjk = sum(1 for c in task if '\u4e00' <= c <= '\u9fff')
        resp_cjk = sum(1 for c in cleaned if '\u4e00' <= c <= '\u9fff')

        task_ratio = task_cjk / max(len(task), 1)
        resp_ratio = resp_cjk / max(len(cleaned), 1)

        # Only flag if task is clearly Chinese and response has almost no Chinese
        if task_ratio > 0.3 and resp_ratio < 0.02:
            return True
        return False

    @staticmethod
    def _check_error_leak(text: str) -> bool:
        """Check for raw Python/Node tracebacks in the response."""
        error_patterns = [
            r"Traceback \(most recent call last\)",
            r"File \"[^\"]+\", line \d+",
            r"raise \w+Error",
            r"node:internal/",
        ]
        return any(re.search(p, text) for p in error_patterns)
