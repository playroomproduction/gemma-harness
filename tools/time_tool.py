"""Time tool — returns current local time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from tools.base import Tool, ToolParameter, register_tool
import config


@register_tool
@dataclass
class GetTimeTool(Tool):
    name: str = "get_time"
    description: str = "Get the current local date and time."
    parameters: list[ToolParameter] = field(default_factory=list)

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        tz = ZoneInfo(config.TIMEZONE)
        now = datetime.now(tz)
        return {
            "timezone": config.TIMEZONE,
            "iso": now.isoformat(timespec="seconds"),
            "human": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "weekday": now.strftime("%A"),
        }
