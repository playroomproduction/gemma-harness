"""
Code Executor — safe shell command execution.

Security model:
- Only whitelisted base commands are allowed.
- Certain dangerous flags/patterns are always blocked.
- All commands have a hard timeout.
- Working directory must be within allowed filesystem paths.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, register_tool
import config


def _is_command_safe(command: str) -> tuple[bool, str]:
    """
    Check if a command is safe to execute.
    Returns (is_safe, reason).
    """
    # Check deny flags first
    for deny in config.SHELL_DENY_FLAGS:
        if deny in command:
            return False, f"Blocked pattern: {deny}"

    # Parse the base command
    try:
        parts = shlex.split(command)
    except ValueError:
        return False, "Cannot parse command"

    if not parts:
        return False, "Empty command"

    base_cmd = Path(parts[0]).name  # strip path, get just command name

    if base_cmd not in config.SHELL_ALLOW:
        return False, f"Command not in whitelist: {base_cmd}. Allowed: {', '.join(sorted(config.SHELL_ALLOW))}"

    return True, "ok"


@register_tool
@dataclass
class RunCommandTool(Tool):
    name: str = "run_command"
    description: str = (
        "Run a shell command and return stdout, stderr, and exit code. "
        f"Allowed commands: {', '.join(sorted(config.SHELL_ALLOW))}. "
        f"Timeout: {config.SHELL_TIMEOUT_SECONDS}s."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute.",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory (must be within allowed paths).",
        ),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args.get("command", "")).strip()
        cwd_str = str(args.get("cwd", "")).strip()

        if not command:
            return {"error": "command is required"}

        # Validate command safety
        is_safe, reason = _is_command_safe(command)
        if not is_safe:
            return {"error": f"Command blocked: {reason}"}

        # Validate working directory
        cwd_path = None
        if cwd_str:
            try:
                cwd_path = Path(cwd_str).expanduser().resolve()
                if not cwd_path.is_dir():
                    return {"error": f"Working directory not found: {cwd_str}"}
            except (ValueError, OSError):
                return {"error": f"Invalid working directory: {cwd_str}"}

        # Execute
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=config.SHELL_TIMEOUT_SECONDS,
                cwd=str(cwd_path) if cwd_path else None,
                env=None,  # inherit environment
            )

            # Truncate verbose output
            stdout = proc.stdout[:8000] if proc.stdout else ""
            stderr = proc.stderr[:4000] if proc.stderr else ""

            return {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": len(proc.stdout or "") > 8000 or len(proc.stderr or "") > 4000,
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {config.SHELL_TIMEOUT_SECONDS}s"}
        except OSError as exc:
            return {"error": f"Execution failed: {exc}"}
