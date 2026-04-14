"""
Agent Loop v2 — Claude-style Perceive → Think → Act → Observe → Reflect.

This is the core engine that turns a basic chat model into
an agentic coding assistant.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from engine.llm_client import LLMClient
from engine.context_manager import ContextManager
from engine.verifier import ResponseVerifier
from tools.base import execute_tool, get_tools_openai_schema, get_tool, get_all_tools

logger = logging.getLogger("gemma-harness.agent")


@dataclass
class AgentResult:
    """Result of an agent loop run."""
    content: str
    tool_calls_made: int = 0
    rounds_used: int = 0
    verification_passed: bool = True
    latency_seconds: float = 0.0
    messages: List[Dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    """
    Claude-style agentic loop with forced planning and verification.

    Architecture:
    1. PERCEIVE: Build context (system prompt + task + memory)
    2. THINK: First LLM call includes planning instruction
    3. ACT: Execute tool calls from the model
    4. OBSERVE: Feed tool results back to the model
    5. REFLECT: Verify final response quality
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        context_mgr: Optional[ContextManager] = None,
        verifier: Optional[ResponseVerifier] = None,
    ):
        self.llm = llm or LLMClient()
        self.context_mgr = context_mgr or ContextManager()
        self.verifier = verifier or ResponseVerifier()
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load the constitution prompt."""
        constitution_path = config.PROMPTS_DIR / "constitution.md"
        guidelines_path = config.PROMPTS_DIR / "tool_guidelines.md"

        prompt_parts = []
        if constitution_path.exists():
            prompt_parts.append(constitution_path.read_text(encoding="utf-8"))
        else:
            prompt_parts.append("You are a helpful coding assistant.")

        if guidelines_path.exists():
            prompt_parts.append(guidelines_path.read_text(encoding="utf-8"))

        full_prompt = "\n\n---\n\n".join(prompt_parts)
        self.context_mgr.set_system_prompt(full_prompt)

    def run(
        self,
        task: str,
        conversation: Optional[List[Dict[str, Any]]] = None,
        max_rounds: Optional[int] = None,
        system_override: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the agentic loop for a given task.

        Args:
            task: The user's request.
            conversation: Optional prior conversation context.
            max_rounds: Override max tool rounds.
            system_override: Override the system prompt.

        Returns:
            AgentResult with the final response and metadata.
        """
        start_time = time.time()
        max_rounds = max_rounds or config.MAX_AGENT_ROUNDS

        if system_override:
            self.context_mgr.set_system_prompt(system_override)

        # ── PERCEIVE ──────────────────────────────────────────────
        memory_keys = self.context_mgr.auto_detect_memory_keys(task)
        if memory_keys:
            logger.info("Auto-loaded memories: %s", memory_keys)

        history = list(conversation or [])
        history.append({"role": "user", "content": task})

        # Smart tool selection: only send relevant tools to reduce prompt size
        tools_schema = self._select_tools(task, history)
        logger.info("Selected %d tools for this task", len(tools_schema))
        total_tool_calls = 0

        # ── THINK + ACT + OBSERVE loop ────────────────────────────
        for round_num in range(1, max_rounds + 1):
            logger.info("Agent round %d/%d", round_num, max_rounds)

            # Build context-aware messages
            messages = self.context_mgr.build_messages(
                history, memory_keys=memory_keys
            )

            # Call LLM
            assistant_msg = self.llm.chat(
                messages=messages,
                tools=tools_schema,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )

            content = assistant_msg.get("content") or ""
            tool_calls = assistant_msg.get("tool_calls") or []

            # Gemma 4 fallback: parse text-based tool calls like
            # <execute_tool>web_search(query="...")</execute_tool>
            if not tool_calls and "<execute_tool>" in content:
                parsed = self._parse_text_tool_calls(content)
                if parsed:
                    tool_calls = parsed
                    logger.info("Parsed %d text-based tool calls from model output", len(parsed))

            # Handle tool hesitation (Gemma 4 common pattern)
            # Instead of nagging the model to call the tool, we call it ourselves
            if not tool_calls and self._is_tool_hesitation(content, history):
                mentioned_tool = self._extract_mentioned_tool(content)
                if mentioned_tool:
                    logger.info("Detected tool hesitation for '%s', auto-calling", mentioned_tool)
                    tool_result = execute_tool(mentioned_tool, {})
                    total_tool_calls += 1

                    # Inject as if the model called it
                    history.append({"role": "assistant", "content": content})
                    history.append({
                        "role": "user",
                        "content": (
                            f"我已經幫你 call 咗 `{mentioned_tool}`，結果如下：\n"
                            f"```json\n{json.dumps(tool_result, ensure_ascii=False, indent=2)[:2000]}\n```\n"
                            "請根據以上結果直接回答用戶嘅問題。"
                        ),
                    })
                    continue
                else:
                    # Can't determine which tool, give a pointed nudge
                    logger.info("Detected tool hesitation but can't identify tool, nudging")
                    history.append({"role": "assistant", "content": content})
                    history.append({
                        "role": "user",
                        "content": "請直接使用工具執行，唔好描述你「會」做咩。",
                    })
                    continue

            # Hallucination guard: detect fake file content without tool call
            if not tool_calls and self._is_file_hallucination(content, history):
                logger.info("Detected file content hallucination, auto-calling read_file")
                # Extract the file path from the original task
                file_path = self._extract_file_path(
                    history[-1].get("content", "") if history else ""
                )
                if file_path:
                    tool_result = execute_tool("read_file", json.dumps({"path": file_path}))
                    total_tool_calls += 1
                    history.append({"role": "assistant", "content": content})
                    history.append({
                        "role": "user",
                        "content": (
                            f"你嘅回覆似乎係估嘅。我已經幫你讀咗真正嘅檔案內容：\n"
                            f"```\n{json.dumps(tool_result, ensure_ascii=False)[:3000]}\n```\n"
                            "請根據以上真實內容重新回答。"
                        ),
                    })
                    continue

            # No tool calls → we have a final response
            if not tool_calls:
                # ── REFLECT ───────────────────────────────────────
                final_content = self._post_process(content, history)

                # Skip verification for error responses
                if final_content.startswith("[ERROR]"):
                    history.append({"role": "assistant", "content": final_content})
                    return AgentResult(
                        content=final_content,
                        tool_calls_made=total_tool_calls,
                        rounds_used=round_num,
                        verification_passed=False,
                        latency_seconds=time.time() - start_time,
                        messages=history,
                    )

                verified = self.verifier.verify(final_content, task, history)

                if not verified.passed and round_num < max_rounds:
                    logger.info(
                        "Verification failed (%s), retrying",
                        verified.reasons,
                    )
                    history.append({"role": "assistant", "content": content})
                    history.append({
                        "role": "user",
                        "content": (
                            f"你嘅回覆有以下問題，請修正後重新回覆：\n"
                            + "\n".join(f"- {r}" for r in verified.reasons)
                        ),
                    })
                    continue

                history.append({"role": "assistant", "content": final_content})
                return AgentResult(
                    content=final_content,
                    tool_calls_made=total_tool_calls,
                    rounds_used=round_num,
                    verification_passed=verified.passed,
                    latency_seconds=time.time() - start_time,
                    messages=history,
                )

            # ── ACT: Execute tool calls ───────────────────────────
            history.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                fn = call.get("function", {})
                tool_name = str(fn.get("name", ""))
                tool_args = fn.get("arguments", "{}")

                # Gemma 4 sometimes uses its own tool names — normalize
                tool_name = self._normalize_tool_name(tool_name)
                tool_args = self._normalize_tool_args(tool_name, tool_args)

                logger.info("Executing tool: %s", tool_name)
                tool_result = execute_tool(tool_name, tool_args)
                total_tool_calls += 1

                history.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False)[:4000],
                })

        # Max rounds exhausted
        logger.warning("Agent exhausted max rounds (%d)", max_rounds)
        return AgentResult(
            content="工具回合已達上限。以下係目前已完成嘅部分：\n\n" + self._summarize_progress(history),
            tool_calls_made=total_tool_calls,
            rounds_used=max_rounds,
            verification_passed=False,
            latency_seconds=time.time() - start_time,
            messages=history,
        )

    @staticmethod
    def _is_tool_hesitation(content: str, messages: list[dict[str, Any]]) -> bool:
        """
        Detect when Gemma 4 describes using a tool instead of actually calling it.
        Common failure mode: "我會用 web_search 工具幫你查..."
        """
        if not content:
            return False
        normalized = content.lower()
        hesitation_phrases = (
            "我會用", "我可以用", "需要先用", "先調用",
            "先獲取", "先查詢", "用工具查", "幫你查",
            "是否需要", "要唔要我", "要不要我",
            "我會幫你", "讓我先",
            "我需要使用", "我需要用", "需要使用",
            "才能告訴", "才能回答",
        )
        tool_mentions = (
            "get_time", "web_search", "fetch_url", "read_file",
            "write_file", "run_command", "search_files",
        )
        has_hesitation = any(p in normalized for p in hesitation_phrases)
        has_tool_mention = any(t in normalized for t in tool_mentions)
        return has_hesitation and has_tool_mention

    @staticmethod
    def _extract_mentioned_tool(content: str) -> Optional[str]:
        """
        Extract the tool name from text where the model describes
        wanting to use a tool.
        """
        normalized = content.lower()
        # Check in priority order (most specific first)
        tool_names = [
            "get_time", "web_search", "fetch_url", "read_file",
            "write_file", "list_directory", "search_files",
            "run_command", "git_status", "git_diff", "git_log",
            "git_checkpoint", "save_memory", "recall_memory",
        ]
        for name in tool_names:
            if name in normalized:
                return name
        return None

    @staticmethod
    def _is_file_hallucination(content: str, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect when the model fabricates file contents instead of calling read_file.
        Pattern: user asks about a file, model returns a code block pretending to
        show file contents, but never actually called read_file.
        """
        if not content:
            return False
        # Check if any previous message asked to read a file
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        task_text = " ".join(user_msgs).lower()
        file_keywords = ("read", "讀", "睇", "file", "檔案", ".py", ".js", ".md", ".json", ".yaml", "config")
        has_file_request = any(kw in task_text for kw in file_keywords)

        # Check if response contains a code block (fake file content)
        has_code_block = "```" in content

        # Check that no read_file tool was actually called
        tool_msgs = [m for m in messages if m.get("role") == "tool" and m.get("name") == "read_file"]
        has_read_file = len(tool_msgs) > 0

        return has_file_request and has_code_block and not has_read_file

    @staticmethod
    def _extract_file_path(text: str) -> Optional[str]:
        """Extract a file path from user message text."""
        # Look for common path patterns
        patterns = [
            r'(~/[^\s\'"`,]+)',           # ~/path/to/file
            r'(/[^\s\'"`,]+\.\w+)',       # /absolute/path.ext
            r'([^\s\'"`,]+\.(?:py|js|ts|md|json|yaml|yml|toml|cfg|conf|txt))',  # file.ext
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    # Gemma 4 built-in tool name → our tool name mapping
    TOOL_NAME_MAP = {
        "google_search": "web_search",
        "search": "web_search",
        "code_execution": "run_command",
        "urlopen": "fetch_url",
    }

    @classmethod
    def _normalize_tool_name(cls, name: str) -> str:
        """Map Gemma 4 built-in tool names to our tool names."""
        return cls.TOOL_NAME_MAP.get(name, name)

    @staticmethod
    def _normalize_tool_args(tool_name: str, raw_args: str) -> str:
        """
        Normalize tool arguments from Gemma 4's conventions to ours.
        E.g. google_search uses {"queries": [...]} but we use {"query": "..."}
        """
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                return raw_args
        else:
            args = raw_args

        if tool_name == "web_search" and "queries" in args and "query" not in args:
            # Gemma 4 sends {"queries": ["..."]} — flatten to {"query": "..."}
            queries = args["queries"]
            if isinstance(queries, list) and queries:
                args = {"query": queries[0]}

        return json.dumps(args, ensure_ascii=False)

    @staticmethod
    def _parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
        """
        Parse Gemma 4's text-based tool calls:
          <execute_tool>
          web_search(query="latest news about Apple")
          </execute_tool>
        """
        calls = []
        # Find all <execute_tool>...</execute_tool> blocks
        for match in re.finditer(
            r"<execute_tool>\s*(\w+)\(([^)]*)\)\s*</execute_tool>",
            content, flags=re.S
        ):
            func_name = match.group(1)
            args_str = match.group(2).strip()

            # Parse key=value pairs
            args = {}
            if args_str:
                # Handle key="value" and key='value' patterns
                for kv_match in re.finditer(
                    r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))',
                    args_str
                ):
                    key = kv_match.group(1)
                    value = kv_match.group(2) or kv_match.group(3) or kv_match.group(4)
                    args[key] = value

            calls.append({
                "id": f"text_call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })

        return calls

    @staticmethod
    def _post_process(content: str, messages: list[dict[str, Any]]) -> str:
        """
        Clean up common Gemma 4 output issues:
        - Strip visible reasoning blocks (UNDERSTAND/PLAN/EXECUTE/VERIFY/RESPOND)
        - Remove meta-commentary
        - Clean up formatting
        """
        cleaned = content.strip()

        # Strip reasoning blocks that should be internal
        reasoning_patterns = [
            r"^\*\*UNDERSTAND\*\*:.*$",
            r"^\*\*PLAN\*\*:.*$",
            r"^\*\*EXECUTE\*\*:.*$",
            r"^\*\*VERIFY\*\*:.*$",
            r"^\*\*RESPOND\*\*:.*$",
        ]
        for pattern in reasoning_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)

        # Remove meta-commentary about the response
        meta_patterns = [
            r"^好的[，,]?\s*",
            r"^沒問題[，,]?\s*",
            r"^OK[，,]?\s*",
            r"^\*?\*?以下係.*回覆[:：]?\*?\*?\s*",
            r"^\*?\*?\[?Issue Comment\s*內容\]?\*?\*?\s*$",
            r"^---+\s*$",
            r"^請提供.*具體任務.*$",
        ]
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)

        # Remove duplicate newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @staticmethod
    def _summarize_progress(messages: list[dict[str, Any]]) -> str:
        """Summarize what was accomplished before hitting the round limit."""
        tool_results = []
        for msg in messages:
            if msg.get("role") == "tool":
                name = msg.get("name", "unknown")
                try:
                    result = json.loads(msg.get("content", "{}"))
                    if "error" in result:
                        tool_results.append(f"- {name}: ❌ {result['error']}")
                    else:
                        tool_results.append(f"- {name}: ✅ 完成")
                except json.JSONDecodeError:
                    tool_results.append(f"- {name}: ⚠️ 結果無法解析")

        if tool_results:
            return "### 已執行工具\n" + "\n".join(tool_results)
        return "未能完成任何工具呼叫。"

    @staticmethod
    def _select_tools(task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select only relevant tools for the current task.
        
        This dramatically reduces prompt size (14 tools → 3-5 tools),
        which is critical for MLX inference speed on 16GB M4.
        """
        task_lower = task.lower()
        all_text = task_lower + " ".join(
            str(m.get("content", "")).lower() for m in messages
        )

        # Always include get_time (tiny schema)
        selected_names = {"get_time"}

        # Keyword → tool mapping
        TOOL_KEYWORDS = {
            "web_search": ["搜尋", "search", "查", "find online", "google", "最新", "news"],
            "fetch_url": ["url", "http", "webpage", "網頁", "link", "連結"],
            "read_file": ["讀", "read", "file", "檔案", "睇", "內容", "code", "config", ".py", ".js", ".md", ".json"],
            "write_file": ["寫", "write", "create", "建", "改", "modify", "新增", "update"],
            "list_directory": ["目錄", "directory", "資料夾", "folder", "ls", "list", "structure"],
            "search_files": ["grep", "搜", "pattern", "codebase", "TODO", "找"],
            "run_command": ["run", "execute", "command", "terminal", "shell", "test", "npm", "pip"],
            "git_status": ["git", "status", "changes", "modified"],
            "git_diff": ["diff", "changed", "改咗"],
            "git_log": ["log", "history", "commit", "記錄"],
            "git_checkpoint": ["checkpoint", "backup", "save", "commit"],
            "save_memory": ["記住", "remember", "save", "store", "memory"],
            "recall_memory": ["recall", "remember", "之前", "memory"],
        }

        for tool_name, keywords in TOOL_KEYWORDS.items():
            if any(kw in all_text for kw in keywords):
                selected_names.add(tool_name)

        # If write_file selected, also include git_checkpoint for safety
        if "write_file" in selected_names:
            selected_names.add("git_checkpoint")

        # If nothing matched, include a small general-purpose set
        if len(selected_names) <= 1:
            selected_names.update(["web_search", "read_file", "list_directory", "run_command"])

        # Cap at 6 tools max to keep prompt manageable
        selected_names_list = list(selected_names)[:6]

        # Build schema for selected tools only
        schema = []
        for tool in get_all_tools():
            if tool.name in selected_names_list:
                schema.append(tool.to_openai_schema())

        return schema
