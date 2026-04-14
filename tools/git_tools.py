"""
Git tools — status, diff, log, checkpoint.

All operations are read-only except git_checkpoint which
creates a safety commit before file modifications.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, register_tool
import config


def _run_git(repo_path: str, git_args: list[str], timeout: int = 15) -> dict[str, Any]:
    """Run a git command in a repo and return the output."""
    try:
        path = Path(repo_path).expanduser().resolve()
    except (ValueError, OSError):
        return {"error": f"Invalid path: {repo_path}"}

    if not (path / ".git").exists() and not (path / ".git").is_file():
        # Check parent dirs for git repo
        check = path
        while check != check.parent:
            if (check / ".git").exists():
                path = check
                break
            check = check.parent
        else:
            return {"error": f"Not a git repository: {repo_path}"}

    cmd = ["git", "-C", str(path)] + git_args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[:8000],
            "stderr": proc.stderr[:2000] if proc.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "Git command timed out"}
    except OSError as exc:
        return {"error": f"Git execution failed: {exc}"}


@register_tool
@dataclass
class GitStatusTool(Tool):
    name: str = "git_status"
    description: str = "Show the working tree status of a git repository (modified, staged, untracked files)."
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="repo_path", type="string", description="Path to the git repository."),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        result = _run_git(str(args.get("repo_path", "")), ["status", "--short", "--branch"])
        if "error" in result:
            return result
        return {"repo": str(args.get("repo_path")), "status": result["stdout"]}


@register_tool
@dataclass
class GitDiffTool(Tool):
    name: str = "git_diff"
    description: str = "Show changes in the working tree or between commits."
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="repo_path", type="string", description="Path to the git repository."),
        ToolParameter(name="file", type="string", description="Specific file to diff (optional).", required=False),
        ToolParameter(name="staged", type="boolean", description="Show staged changes only (default false).", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        git_args = ["diff"]
        if args.get("staged"):
            git_args.append("--cached")
        git_args.append("--stat")  # summary first
        file_path = args.get("file")
        if file_path:
            git_args.extend(["--", str(file_path)])

        result = _run_git(str(args.get("repo_path", "")), git_args)
        if "error" in result:
            return result

        # Also get the actual diff (limited)
        detail_args = ["diff"]
        if args.get("staged"):
            detail_args.append("--cached")
        if file_path:
            detail_args.extend(["--", str(file_path)])

        detail = _run_git(str(args.get("repo_path", "")), detail_args)
        diff_text = detail.get("stdout", "")[:6000]

        return {
            "repo": str(args.get("repo_path")),
            "summary": result["stdout"],
            "diff": diff_text,
        }


@register_tool
@dataclass
class GitLogTool(Tool):
    name: str = "git_log"
    description: str = "Show recent commit history."
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="repo_path", type="string", description="Path to the git repository."),
        ToolParameter(name="n", type="integer", description="Number of commits to show (default 5).", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        n = min(int(args.get("n") or 5), 20)
        result = _run_git(
            str(args.get("repo_path", "")),
            ["log", f"-{n}", "--oneline", "--no-decorate"],
        )
        if "error" in result:
            return result
        return {"repo": str(args.get("repo_path")), "log": result["stdout"]}


@register_tool
@dataclass
class GitCheckpointTool(Tool):
    name: str = "git_checkpoint"
    description: str = (
        "Create a safety commit with all current changes. "
        "Use this BEFORE making file modifications to enable easy rollback."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="repo_path", type="string", description="Path to the git repository."),
        ToolParameter(name="message", type="string", description="Checkpoint commit message."),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        repo = str(args.get("repo_path", ""))
        msg = str(args.get("message", "gemma-harness checkpoint"))

        # Stage all changes
        add_result = _run_git(repo, ["add", "-A"])
        if "error" in add_result:
            return add_result

        # Check if there's anything to commit
        status = _run_git(repo, ["status", "--porcelain"])
        if not status.get("stdout", "").strip():
            return {"ok": True, "message": "Nothing to commit, working tree clean"}

        # Commit
        commit_result = _run_git(repo, ["commit", "-m", f"[checkpoint] {msg}"])
        if "error" in commit_result:
            return commit_result

        return {
            "ok": commit_result.get("exit_code", 1) == 0,
            "message": commit_result.get("stdout", ""),
        }
