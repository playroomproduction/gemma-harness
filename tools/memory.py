"""
Memory tool — persistent key-value memory across sessions.

Stores memories as individual JSON files in ~/.gemma-harness/memory/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, register_tool
import config


def _memory_path(key: str) -> Path:
    """Get the file path for a memory key."""
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return config.MEMORY_DIR / f"{safe_key}.json"


@register_tool
@dataclass
class SaveMemoryTool(Tool):
    name: str = "save_memory"
    description: str = (
        "Save information to persistent memory. "
        "Use for decisions, preferences, discoveries that should persist across sessions."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="key", type="string", description="Short identifier for the memory (e.g., 'project_structure', 'user_preferences')."),
        ToolParameter(name="content", type="string", description="The information to remember."),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("key", "")).strip()
        content = str(args.get("content", "")).strip()

        if not key:
            return {"error": "key is required"}
        if not content:
            return {"error": "content is required"}
        if len(content) > 10000:
            return {"error": "content too long (max 10,000 chars)"}

        path = _memory_path(key)
        data = {
            "key": key,
            "content": content,
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        }

        # If already exists, preserve created timestamp
        if path.exists():
            try:
                existing = json.loads(path.read_text())
                data["created"] = existing.get("created", data["created"])
            except (json.JSONDecodeError, OSError):
                pass

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return {"ok": True, "key": key}


@register_tool
@dataclass
class RecallMemoryTool(Tool):
    name: str = "recall_memory"
    description: str = (
        "Retrieve information from persistent memory. "
        "Call with no key to list all stored memories."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="key", type="string", description="The memory key to retrieve. Omit to list all.", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("key", "")).strip() if args.get("key") else ""

        if key:
            path = _memory_path(key)
            if not path.exists():
                return {"error": f"No memory found for key: {key}"}
            try:
                data = json.loads(path.read_text())
                return {"key": key, "content": data["content"], "updated": data.get("updated")}
            except (json.JSONDecodeError, OSError) as exc:
                return {"error": f"Cannot read memory: {exc}"}

        # List all memories
        memories = []
        for f in sorted(config.MEMORY_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                memories.append({
                    "key": data.get("key", f.stem),
                    "preview": data.get("content", "")[:100],
                    "updated": data.get("updated"),
                })
            except (json.JSONDecodeError, OSError):
                continue

        return {"count": len(memories), "memories": memories}
