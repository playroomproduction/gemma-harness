"""Web tools — search and fetch URL content."""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from tools.base import Tool, ToolParameter, register_tool


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


@register_tool
@dataclass
class WebSearchTool(Tool):
    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo and return top result titles and URLs."
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(
            name="query",
            type="string",
            description="The search query.",
        ),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}

        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")

        results = []
        for match in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.I | re.S,
        ):
            href = html.unescape(match.group(1))
            title = _strip_tags(match.group(2))
            if title and href:
                results.append({"title": title, "url": href})
            if len(results) >= 8:
                break

        return {"query": query, "results": results}


@register_tool
@dataclass
class FetchUrlTool(Tool):
    name: str = "fetch_url"
    description: str = "Fetch a URL and return a plain-text extract (max 12,000 chars)."
    parameters: list[ToolParameter] = field(default_factory=lambda: [
        ToolParameter(
            name="url",
            type="string",
            description="Absolute http(s) URL to fetch.",
        ),
    ])

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            return {"error": "url must start with http:// or https://"}

        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(300_000).decode("utf-8", "replace")

        return {"url": url, "text": _strip_tags(raw)[:12000]}
