"""
Gemma Harness Server — FastAPI service on port 8093.

Endpoints:
  POST /chat       — Interactive chat with full harness
  POST /invoke     — Backward-compatible Paperclip interface
  GET  /health     — Health check
  GET  /tools      — List registered tools
  GET  /memory     — View stored memories
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

import config

# Force tool registration before anything else
import tools  # noqa: F401

from engine.agent_loop import AgentLoop
from engine.llm_client import LLMClient
from tools.base import get_all_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gemma-harness.server")

# ── App Setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="Gemma Harness",
    description="Claude-style agentic harness for local Gemma 4 E4B",
    version="0.1.0",
)

# Shared instances
_llm = LLMClient()
_loaded_model = _llm.detect_model()
logger.info("Detected MLX model: %s", _loaded_model)
_agent = AgentLoop(llm=_llm)


# ── Request/Response Models ─────────────────────────────────────────
class ChatRequest(BaseModel):
    """Interactive chat request."""
    message: str
    conversation: Optional[List[Dict[str, Any]]] = None
    max_rounds: Optional[int] = None
    system: Optional[str] = None


class InvokeRequest(BaseModel):
    """Backward-compatible Paperclip bridge request."""
    model_config = ConfigDict(extra="allow")

    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    system: Optional[str] = None
    max_tool_rounds: int = 4
    writeback: bool = False
    agentId: Optional[str] = None
    runId: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# ── Endpoints ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — also verifies MLX server connectivity."""
    mlx_ok = _llm.health_check()
    tool_count = len(get_all_tools())
    return {
        "ok": mlx_ok,
        "model": config.MLX_MODEL,
        "mlx_server": config.MLX_BASE_URL,
        "mlx_healthy": mlx_ok,
        "tools_registered": tool_count,
        "harness_version": "0.1.0",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Interactive chat with full agentic harness.
    This is the primary endpoint for enhanced Gemma 4 interaction.
    """
    logger.info("Chat request: %s", req.message[:100])

    result = _agent.run(
        task=req.message,
        conversation=req.conversation,
        max_rounds=req.max_rounds,
        system_override=req.system,
    )

    return {
        "ok": True,
        "content": result.content,
        "model": config.MLX_MODEL,
        "tool_calls_made": result.tool_calls_made,
        "rounds_used": result.rounds_used,
        "verification_passed": result.verification_passed,
        "latency_seconds": round(result.latency_seconds, 2),
    }


@app.post("/invoke")
def invoke(http_request: Request, req: InvokeRequest):
    """
    Backward-compatible Paperclip bridge interface.
    Translates legacy request format to the new harness.
    """
    logger.info("Invoke request (legacy bridge compat)")

    # Build task from legacy format
    if req.messages:
        # Extract the last user message as the task
        user_msgs = [m for m in req.messages if m.get("role") == "user"]
        task = user_msgs[-1]["content"] if user_msgs else "請根據 context 回覆。"
        conversation = req.messages[:-1] if len(req.messages) > 1 else None
    elif req.prompt:
        task = req.prompt
        conversation = None
    elif req.context:
        task = json.dumps(req.context, ensure_ascii=False, indent=2)
        conversation = None
    else:
        task = "請根據 context 回覆。"
        conversation = None

    result = _agent.run(
        task=task,
        conversation=conversation,
        max_rounds=req.max_tool_rounds,
        system_override=req.system,
    )

    return {
        "ok": True,
        "model": config.MLX_MODEL,
        "content": result.content,
        "issueId": None,
        "writeback": None,
        "tool_calls_made": result.tool_calls_made,
    }


@app.get("/tools")
def list_tools():
    """List all registered tools with their schemas."""
    tools_list = get_all_tools()
    return {
        "count": len(tools_list),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [
                    {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                    for p in t.parameters
                ],
            }
            for t in tools_list
        ],
    }


@app.get("/memory")
def list_memory():
    """View all stored memories."""
    memories = []
    for f in sorted(config.MEMORY_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            memories.append({
                "key": data.get("key", f.stem),
                "content": data.get("content", "")[:200],
                "updated": data.get("updated"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return {"count": len(memories), "memories": memories}


# ── Error Handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
    )


# ── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting Gemma Harness on %s:%d (model: %s)",
        config.HARNESS_HOST,
        config.HARNESS_PORT,
        config.MLX_MODEL,
    )
    uvicorn.run(app, host=config.HARNESS_HOST, port=config.HARNESS_PORT)
