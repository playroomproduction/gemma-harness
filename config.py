"""
Gemma Harness — Central Configuration

All security boundaries, whitelists, and runtime settings live here.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── MLX Server ──────────────────────────────────────────────────────
MLX_BASE_URL = os.getenv("GEMMA_HARNESS_MLX_URL", "http://127.0.0.1:8091")
MLX_CHAT_ENDPOINT = f"{MLX_BASE_URL}/v1/chat/completions"
MLX_MODEL = os.getenv("GEMMA_HARNESS_MLX_MODEL", "mlx-community/gemma-4-e4b-it-4bit")

# ── Harness Server ──────────────────────────────────────────────────
HARNESS_HOST = os.getenv("GEMMA_HARNESS_HOST", "127.0.0.1")
HARNESS_PORT = int(os.getenv("GEMMA_HARNESS_PORT", "8093"))

# ── Timezone ────────────────────────────────────────────────────────
TIMEZONE = os.getenv("GEMMA_HARNESS_TZ", "Europe/London")

# ── Agent Loop ──────────────────────────────────────────────────────
MAX_AGENT_ROUNDS = int(os.getenv("GEMMA_HARNESS_MAX_ROUNDS", "8"))
MAX_TOKENS = int(os.getenv("GEMMA_HARNESS_MAX_TOKENS", "1536"))
TEMPERATURE = float(os.getenv("GEMMA_HARNESS_TEMPERATURE", "0"))
LLM_TIMEOUT = float(os.getenv("GEMMA_HARNESS_LLM_TIMEOUT", "300"))

# ── Context Budget ──────────────────────────────────────────────────
# Leave headroom for generation; Gemma 4 E4B supports 128k context
# but on 16GB M4 we must be conservative with KV cache.
MAX_CONTEXT_TOKENS = int(os.getenv("GEMMA_HARNESS_CONTEXT_BUDGET", "18000"))

# ── Filesystem Security ────────────────────────────────────────────
# Directories the agent is allowed to read.
FS_READ_ALLOW = [
    Path.home() / "Documents" / "Coding",
    Path.home() / "playroom-library",
    Path.home() / ".hermes",
    Path.home() / "dashboard",
]

# Directories the agent is allowed to write (subset of read).
FS_WRITE_ALLOW = [
    Path.home() / "Documents" / "Coding",
]

# Patterns to never read or write.
FS_DENY_PATTERNS = [
    ".env",
    ".git/objects",
    "node_modules",
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "*.key",
    "*.pem",
    "*.p12",
    "id_rsa",
    "id_ed25519",
]

# ── Shell Security ──────────────────────────────────────────────────
# Commands the agent is allowed to run directly.
SHELL_ALLOW = {
    "ls", "cat", "head", "tail", "wc",
    "grep", "rg", "find", "fd",
    "python3", "node",
    "git",
    "echo", "date", "pwd",
    "curl",  # read-only web requests
    "jq",
}

# Subcommands / flags that are always blocked, even for allowed commands.
SHELL_DENY_FLAGS = {
    "rm -rf", "rm -r /",
    "sudo",
    "chmod 777",
    "mkfs",
    "> /dev/",
    "| sh", "| bash",
    "eval",
    "exec",
}

SHELL_TIMEOUT_SECONDS = 30

# ── Memory ──────────────────────────────────────────────────────────
MEMORY_DIR = Path.home() / ".gemma-harness" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompts ─────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent / "prompts"
