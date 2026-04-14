"""
Tool Base Class & Registry

All tools follow the same pattern:
1. Declare as a dataclass with name, description, parameters (JSON Schema)
2. Implement execute() → dict
3. Register via @register_tool decorator

The registry auto-generates the OpenAI-compatible tools array
for the MLX server.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gemma-harness.tools")

# Global tool registry
_REGISTRY: dict[str, "Tool"] = {}


@dataclass
class ToolParameter:
    """A single parameter in a tool's JSON Schema."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class Tool(ABC):
    """Base class for all harness tools."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the tool and return a result dict."""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible tool definition."""
        properties = {}
        required = []
        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


def register_tool(tool_or_cls):
    """
    Register a tool in the global registry.

    Can be used as:
      @register_tool        — on a @dataclass class (auto-instantiates)
      register_tool(inst)   — with a pre-built instance
    """
    # If it's a class, instantiate it first
    if isinstance(tool_or_cls, type) and issubclass(tool_or_cls, Tool):
        tool = tool_or_cls()
        if tool.name in _REGISTRY:
            logger.warning("Tool %s already registered, overwriting", tool.name)
        _REGISTRY[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)
        return tool_or_cls  # Return the class so it can still be used normally
    elif isinstance(tool_or_cls, Tool):
        tool = tool_or_cls
        if tool.name in _REGISTRY:
            logger.warning("Tool %s already registered, overwriting", tool.name)
        _REGISTRY[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)
        return tool
    else:
        raise TypeError(f"register_tool expects a Tool class or instance, got {type(tool_or_cls)}")


def get_tool(name: str) -> Optional[Tool]:
    """Look up a tool by name."""
    return _REGISTRY.get(name)


def get_all_tools() -> list[Tool]:
    """Return all registered tools."""
    return list(_REGISTRY.values())


def get_tools_openai_schema() -> list[dict[str, Any]]:
    """Return all tools as OpenAI-compatible JSON array."""
    return [tool.to_openai_schema() for tool in _REGISTRY.values()]


def execute_tool(name: str, raw_args: 'str | dict') -> Dict[str, Any]:
    """
    Look up and execute a tool by name.
    Handles JSON parsing and error wrapping.
    """
    tool = get_tool(name)
    if not tool:
        return {"error": f"Unknown tool: {name}"}

    # Parse arguments
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON arguments for {name}: {raw_args[:200]}"}
    else:
        args = raw_args

    # Execute with error handling
    try:
        result = tool.execute(args)
        logger.info("Tool %s executed successfully", name)
        return result
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
