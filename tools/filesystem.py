"""
Filesystem tools — read, write, list, search files.

All operations are sandboxed to configured allowed directories.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, register_tool
import config


def _resolve_and_check(path_str: str, allow_list: list[Path]) -> Path | None:
    """
    Resolve a path and check it against the allow list.
    Returns the resolved Path if allowed, None otherwise.
    """
    try:
        resolved = Path(path_str).expanduser().resolve()
    except (ValueError, OSError):
        return None

    # Check against deny patterns
    for pattern in config.FS_DENY_PATTERNS:
        if fnmatch.fnmatch(resolved.name, pattern):
            return None
        # Also check parent components
        for part in resolved.parts:
            if fnmatch.fnmatch(part, pattern):
                return None

    # Check against allow list
    for allowed in allow_list:
        allowed_resolved = allowed.resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue

    return None


@register_tool
@dataclass
class ReadFileTool(Tool):
    name: str = "read_file"
    description: str = (
        "Read file contents. Specify start_line/end_line for partial reads. "
        "Returns at most 500 lines per call to conserve context."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="path", type="string", description="Absolute or ~ path to the file."),
        ToolParameter(name="start_line", type="integer", description="1-indexed start line (optional).", required=False),
        ToolParameter(name="end_line", type="integer", description="1-indexed end line, inclusive (optional).", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_and_check(str(args.get("path", "")), config.FS_READ_ALLOW)
        if path is None:
            return {"error": f"Path not allowed or not found: {args.get('path')}"}
        if not path.is_file():
            return {"error": f"Not a file: {path}"}

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"error": f"Cannot read {path}: {exc}"}

        lines = content.splitlines()
        total = len(lines)
        start = max(1, int(args.get("start_line") or 1))
        end = min(total, int(args.get("end_line") or total))

        # Cap at 500 lines
        if end - start + 1 > 500:
            end = start + 499

        selected = lines[start - 1: end]
        return {
            "path": str(path),
            "total_lines": total,
            "showing": f"{start}-{end}",
            "content": "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=start)),
        }


@register_tool
@dataclass
class WriteFileTool(Tool):
    name: str = "write_file"
    description: str = (
        "Write content to a file. Creates parent directories if needed. "
        "Only works within allowed write directories."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="path", type="string", description="Absolute or ~ path to write."),
        ToolParameter(name="content", type="string", description="The full file content to write."),
        ToolParameter(name="create_parents", type="boolean", description="Create parent dirs if missing.", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(args.get("path", ""))
        path = _resolve_and_check(raw_path, config.FS_WRITE_ALLOW)
        if path is None:
            return {"error": f"Write not allowed: {raw_path}"}

        content = str(args.get("content", ""))
        if args.get("create_parents", True):
            path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"error": f"Cannot write {path}: {exc}"}

        return {"ok": True, "path": str(path), "bytes_written": len(content.encode("utf-8"))}


@register_tool
@dataclass
class ListDirectoryTool(Tool):
    name: str = "list_directory"
    description: str = "List files and subdirectories in a directory."
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="path", type="string", description="Absolute or ~ path to the directory."),
        ToolParameter(name="max_depth", type="integer", description="Max recursion depth (default 1).", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_and_check(str(args.get("path", "")), config.FS_READ_ALLOW)
        if path is None:
            return {"error": f"Path not allowed: {args.get('path')}"}
        if not path.is_dir():
            return {"error": f"Not a directory: {path}"}

        max_depth = int(args.get("max_depth") or 1)
        entries: list[dict[str, Any]] = []
        self._walk(path, path, 0, max_depth, entries)

        return {"path": str(path), "entries": entries[:200]}  # Cap at 200

    @staticmethod
    def _walk(root: Path, current: Path, depth: int, max_depth: int, out: list) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for child in children:
            # Skip denied patterns
            skip = False
            for pattern in config.FS_DENY_PATTERNS:
                if fnmatch.fnmatch(child.name, pattern):
                    skip = True
                    break
            if skip:
                continue

            rel = str(child.relative_to(root))
            if child.is_dir():
                out.append({"name": rel + "/", "type": "dir"})
                if depth < max_depth:
                    ListDirectoryTool._walk(root, child, depth + 1, max_depth, out)
            else:
                size = child.stat().st_size
                out.append({"name": rel, "type": "file", "size_bytes": size})


@register_tool
@dataclass
class SearchFilesTool(Tool):
    name: str = "search_files"
    description: str = (
        "Search for a text pattern across files in a directory (like grep). "
        "Returns matching lines with file paths and line numbers."
    )
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(name="path", type="string", description="Directory to search in."),
        ToolParameter(name="pattern", type="string", description="Text or regex pattern to search for."),
        ToolParameter(name="file_glob", type="string", description="File glob filter, e.g. '*.py' (optional).", required=False),
        ToolParameter(name="case_insensitive", type="boolean", description="Case-insensitive search (default true).", required=False),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_and_check(str(args.get("path", "")), config.FS_READ_ALLOW)
        if path is None:
            return {"error": f"Path not allowed: {args.get('path')}"}

        pattern = str(args.get("pattern", ""))
        if not pattern:
            return {"error": "pattern is required"}

        # Use ripgrep if available, fall back to grep
        cmd = ["rg", "--json", "-m", "30"]
        if args.get("case_insensitive", True):
            cmd.append("-i")
        if args.get("file_glob"):
            cmd.extend(["-g", str(args["file_glob"])])
        cmd.extend([pattern, str(path)])

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15, cwd=str(path)
            )
        except FileNotFoundError:
            # ripgrep not installed, fall back to grep
            cmd = ["grep", "-rn", "--include=" + str(args.get("file_glob", "*")),
                   "-m", "30"]
            if args.get("case_insensitive", True):
                cmd.append("-i")
            cmd.extend([pattern, str(path)])
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=15, cwd=str(path)
                )
            except Exception as exc:
                return {"error": f"Search failed: {exc}"}
        except subprocess.TimeoutExpired:
            return {"error": "Search timed out after 15 seconds"}

        # Parse ripgrep JSON output
        matches = []
        for line in proc.stdout.splitlines()[:50]:
            try:
                obj = __import__("json").loads(line)
                if obj.get("type") == "match":
                    data = obj["data"]
                    matches.append({
                        "file": data["path"]["text"],
                        "line": data["line_number"],
                        "text": data["lines"]["text"].rstrip(),
                    })
            except (ValueError, KeyError):
                # Grep-style output: file:line:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    matches.append({
                        "file": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "text": parts[2].rstrip(),
                    })

        return {"pattern": pattern, "match_count": len(matches), "matches": matches}
