"""
LLM Client — communicates with the local MLX server.

Uses httpx for async support and provides both sync and
async interfaces for the MLX chat completions endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

import config

logger = logging.getLogger("gemma-harness.llm")


class LLMClient:
    """Synchronous client for the local MLX server."""

    def __init__(
        self,
        chat_url: str = config.MLX_CHAT_ENDPOINT,
        model: str = config.MLX_MODEL,
        timeout: float = config.LLM_TIMEOUT,
    ):
        self.chat_url = chat_url
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = config.TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
    ) -> dict[str, Any]:
        """
        Send a chat completion request to the MLX server.
        Returns the assistant message dict with 'content' and optional 'tool_calls'.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        logger.debug(
            "LLM request: %d messages, %d tools",
            len(messages),
            len(tools) if tools else 0,
        )

        try:
            resp = self._client.post(self.chat_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            logger.error("MLX server request timed out after %.0fs", self.timeout)
            return {"content": "[ERROR] MLX server request timed out.", "tool_calls": []}
        except httpx.HTTPStatusError as exc:
            logger.error("MLX server HTTP error: %s", exc)
            return {"content": f"[ERROR] MLX server error: {exc.response.status_code}", "tool_calls": []}
        except httpx.ConnectError:
            logger.error("Cannot connect to MLX server at %s", self.chat_url)
            return {"content": "[ERROR] Cannot connect to MLX server. Is it running?", "tool_calls": []}

        try:
            message = data["choices"][0]["message"]
            logger.debug(
                "LLM response: content=%d chars, tool_calls=%d",
                len(message.get("content") or ""),
                len(message.get("tool_calls") or []),
            )
            return message
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected MLX response structure: %s", exc)
            return {"content": f"[ERROR] Unexpected response: {json.dumps(data)[:500]}", "tool_calls": []}

    def health_check(self) -> bool:
        """Check if the MLX server is healthy."""
        try:
            resp = self._client.get(config.MLX_BASE_URL + "/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def detect_model(self) -> str:
        """Auto-detect which model the MLX server has loaded."""
        try:
            resp = self._client.get(config.MLX_BASE_URL + "/health", timeout=5.0)
            data = resp.json()
            loaded = data.get("model_name", self.model)
            if loaded != self.model:
                logger.info("MLX server has %s loaded (config says %s), using loaded model", loaded, self.model)
                self.model = loaded
            return loaded
        except Exception:
            return self.model

    def close(self):
        self._client.close()
