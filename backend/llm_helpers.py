"""Shared helpers for Anthropic Claude calls.

Centralizes the web search tool definition so every call site uses the same
config, and provides a tolerant response parser that handles the multi-block
content shape you get when server tools are enabled.

Why a shared module instead of inlining: the web search tool config
(max_uses, user_location, allowed_domains) is the kind of thing you want to
change in exactly one place. Same for the response parser — when tools
are enabled Claude emits `text`, `server_tool_use`, `web_search_tool_result`,
and more `text` blocks, and callers that do `message.content[0].text`
will silently break the moment a tool block lands in index 0.
"""

from __future__ import annotations

import re
from typing import Any, Dict


# Localize searches for Pakistan Stock Exchange context — results will favor
# Pakistan and regional sources first.
_PK_LOCATION: Dict[str, Any] = {
    "type": "approximate",
    "city": "Karachi",
    "region": "Sindh",
    "country": "PK",
    "timezone": "Asia/Karachi",
}


# Web search tool for stock-level sentiment analysis. max_uses is tight
# because each sentiment call runs per ticker — latency and cost add up.
WEB_SEARCH_TOOL_SENTIMENT: Dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 2,
    "user_location": _PK_LOCATION,
}


# Web search tool for geopolitical trajectory assessment. Gets a larger
# budget because one analysis may verify multiple conflicts / macro events.
WEB_SEARCH_TOOL_GEO: Dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 3,
    "user_location": _PK_LOCATION,
}


def extract_final_text(message: Any) -> str:
    """Pull the final assistant text block out of a Claude Messages response.

    With server tools enabled the response `content` is a list of mixed
    blocks (text, server_tool_use, web_search_tool_result, ...). The final
    answer is the last `text` block — earlier text blocks are narration
    Claude emits while deciding to search.

    Raises ValueError if no text content is present (e.g. tool errored out
    and the assistant produced nothing).
    """
    content = getattr(message, "content", None) or []
    text_blocks: list[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", None)
            if text:
                text_blocks.append(text)
    if not text_blocks:
        raise ValueError("Claude response has no text content blocks")
    return text_blocks[-1].strip()


def extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` JSON object found in ``text``.

    Tolerant to Claude's occasional preamble ("Based on my research...")
    and markdown code fences. Falls back to returning the stripped text
    unchanged if no brace pair is found, so downstream ``json.loads`` can
    still raise its usual error with the original content.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()

    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return s


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_claude_json_response(message: Any) -> str:
    """Convenience: final text → JSON object string, ready for ``json.loads``."""
    text = extract_final_text(message)
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    return extract_json_object(text)
