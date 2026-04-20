"""
Offline CacheBlend-style trace analysis and visualization.

Reads raw JSONL traces (Anthropic or OpenClaw/OpenAI format), groups requests
by session, and computes prefix / non-prefix / new-compute token reuse metrics
with per-block visualization metadata.

Usage:
    python offline_analysis/analyze_trace.py <trace.jsonl>
    python offline_analysis/analyze_trace.py <trace.jsonl> --format openclaw
    python offline_analysis/analyze_trace.py <trace.jsonl> --html output.html
    python offline_analysis/analyze_trace.py <trace.jsonl> --format openclaw --html output.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import tiktoken

from cacheblend_hashes import (
    DEFAULT_CHUNK_SIZE,
    BlendMatchResult,
    BlendTokenRangeMatcher,
    RollingPrefixHasher,
    _poly_hash,
    chunk_fingerprint_id as _make_fingerprint_id,
    storage_hash_id,
)


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_TIKTOKEN_ENCODING = "o200k_base"


# ── Agent type detection ──────────────────────────────────────────────────

def _detect_agent_type(body: dict[str, Any]) -> str:
    """Classify a request as main agent, subagent, compact, or init.

    Detection uses the tool set as fingerprint:
    - Main agent: has the "Agent" tool (can spawn subagents)
    - Subagent: has tools but no "Agent" tool
    - Compact: only 0-2 tools (compaction/summary agent)
    - Init: no tools at all (initialization request)
    """
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return "no_tools"
    tool_names = {t.get("name", "") for t in tools if isinstance(t, dict)}
    if len(tool_names) <= 2:
        return "compact"
    if "Agent" in tool_names:
        return "main"
    return "subagent"


def _detect_agent_type_openclaw(body: dict[str, Any]) -> str:
    """Classify an OpenClaw request — simplified since there's no Agent tool.

    OpenClaw sessions are single-agent with accumulating history.
    """
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        return "with_tools"
    return "session"


def _compute_tool_fingerprint_openclaw(body: dict[str, Any]) -> str:
    """Hash of sorted tool function names in OpenAI function-call format."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return "no_tools"
    names = sorted(
        t.get("function", {}).get("name", "")
        for t in tools
        if isinstance(t, dict)
    )
    return "|".join(names)


def _compute_tool_fingerprint(body: dict[str, Any]) -> str:
    """Hash of sorted tool names — identifies distinct agent configurations."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return "no_tools"
    names = sorted(t.get("name", "") for t in tools if isinstance(t, dict))
    return "|".join(names)


def _extract_agent_handoffs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract Agent tool handoff data (spawn prompt + return result) from messages."""
    handoffs: list[dict[str, Any]] = []
    # Pass 1: collect Agent tool_use blocks
    agent_tool_uses: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "Agent":
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    prompt_text = str(inp.get("prompt", ""))
                    agent_tool_uses[block.get("id", "")] = {
                        "prompt_preview": prompt_text[:500],
                        "prompt_length": len(prompt_text),
                        "description": str(inp.get("description", "")),
                        "subagent_type": str(inp.get("subagent_type", "")),
                    }
    # Pass 2: match tool_results to Agent tool_uses
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tu_id = block.get("tool_use_id", "")
                if tu_id in agent_tool_uses:
                    tu_info = agent_tool_uses.pop(tu_id)
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_text = " ".join(
                            b.get("text", "")
                            for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    elif isinstance(result_content, str):
                        result_text = result_content
                    else:
                        result_text = str(result_content)
                    handoffs.append({
                        "prompt_preview": tu_info["prompt_preview"],
                        "prompt_length": tu_info["prompt_length"],
                        "description": tu_info["description"],
                        "subagent_type": tu_info["subagent_type"],
                        "result_preview": result_text[:500],
                        "result_length": len(result_text),
                    })
    return handoffs


# ── Enums ──────────────────────────────────────────────────────────────────

class CacheSourceMode(str, Enum):
    REQUEST_ONLY = "request_only"
    INCLUDE_DECODE_CACHE = "include_decode_cache"


# ── Prompt reconstruction ──────────────────────────────────────────────────

def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _normalize_content(
    content: Any,
    *,
    strip_assistant_thinking: bool,
    role: str,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        blocks: list[Any] = [_normalize_text_block(content)]
    elif isinstance(content, list):
        blocks = content[:]
    elif content is None:
        blocks = []
    else:
        blocks = [_normalize_text_block(str(content))]

    normalized: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, str):
            block = _normalize_text_block(block)
        elif not isinstance(block, dict):
            block = {"type": "text", "text": str(block)}

        if (
            strip_assistant_thinking
            and role == "assistant"
            and block.get("type") == "thinking"
        ):
            continue
        normalized.append(block)
    return normalized


def _normalize_system(
    system: Any,
    *,
    strip_billing_header: bool = True,
) -> list[dict[str, Any]]:
    def _keep_system_block(block: dict[str, Any]) -> bool:
        if strip_billing_header:
            text = block.get("text")
            if isinstance(text, str) and text.startswith("x-anthropic-billing-header:"):
                return False
        return True

    if isinstance(system, str):
        block = _normalize_text_block(system)
        return [block] if _keep_system_block(block) else []
    if isinstance(system, list):
        result: list[dict[str, Any]] = []
        for item in system:
            if isinstance(item, str):
                block = _normalize_text_block(item)
            elif isinstance(item, dict):
                block = item
            else:
                block = _normalize_text_block(str(item))
            if _keep_system_block(block):
                result.append(block)
        return result
    if system is None:
        return []
    block = _normalize_text_block(str(system))
    return [block] if _keep_system_block(block) else []


def build_prompt_structure(
    body: dict[str, Any],
    *,
    strip_assistant_thinking: bool = True,
    strip_billing_header: bool = True,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {}

    system = _normalize_system(body.get("system"), strip_billing_header=strip_billing_header)
    if system:
        prompt["system"] = system

    messages: list[dict[str, Any]] = []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        normalized_message = {
            "role": role,
            "content": _normalize_content(
                message.get("content"),
                strip_assistant_thinking=strip_assistant_thinking,
                role=role,
            ),
        }
        for key, value in message.items():
            if key in {"role", "content"}:
                continue
            normalized_message[key] = value
        messages.append(normalized_message)

    prompt["messages"] = messages

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        prompt["tools"] = tools

    return prompt


def build_prompt_structure_openclaw(
    body: dict[str, Any],
    *,
    strip_assistant_thinking: bool = True,
) -> dict[str, Any]:
    """Build prompt structure for OpenClaw/OpenAI format traces.

    In OpenClaw format, system is messages[0] with role="system" — there is
    no top-level 'system' field.  Tools (if any) use OpenAI function format.
    """
    prompt: dict[str, Any] = {}

    messages: list[dict[str, Any]] = []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        normalized_message = {
            "role": role,
            "content": _normalize_content(
                message.get("content"),
                strip_assistant_thinking=strip_assistant_thinking,
                role=role,
            ),
        }
        for key, value in message.items():
            if key in {"role", "content"}:
                continue
            normalized_message[key] = value
        messages.append(normalized_message)

    prompt["messages"] = messages

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        prompt["tools"] = tools

    return prompt


def _serialize_prompt_sections(prompt: dict[str, Any]) -> list[str]:
    sections: list[str] = []

    tools = prompt.get("tools")
    if tools:
        sections.append(_stable_json({"section": "tools", "tools": tools}))

    system = prompt.get("system")
    if system:
        sections.append(_stable_json({"section": "system", "blocks": system}))

    for message in prompt.get("messages", []):
        sections.append(_stable_json({"section": "message", "message": message}))

    return sections


def _serialize_prompt_sections_openclaw(prompt: dict[str, Any]) -> list[str]:
    """Serialize OpenClaw prompt: messages only (system is already messages[0]).

    If tools are present, they come before messages.
    """
    sections: list[str] = []

    tools = prompt.get("tools")
    if tools:
        sections.append(_stable_json({"section": "tools", "tools": tools}))

    for message in prompt.get("messages", []):
        sections.append(_stable_json({"section": "message", "message": message}))

    return sections


def build_assistant_response_message(
    body: dict[str, Any],
    *,
    strip_assistant_thinking: bool = True,
) -> dict[str, Any] | None:
    role = str(body.get("role", "assistant"))
    if role != "assistant":
        return None

    content = _normalize_content(
        body.get("content"),
        strip_assistant_thinking=strip_assistant_thinking,
        role="assistant",
    )
    if not content:
        return None

    message: dict[str, Any] = {"role": "assistant", "content": content}
    for key in ("stop_reason", "stop_sequence", "model"):
        if key in body:
            message[key] = body[key]
    return message


def serialize_assistant_response_sections(
    body: dict[str, Any],
    *,
    strip_assistant_thinking: bool = True,
) -> list[str]:
    message = build_assistant_response_message(
        body, strip_assistant_thinking=strip_assistant_thinking,
    )
    if message is None:
        return []
    return [_stable_json({"section": "message", "message": message})]


def serialize_assistant_response_sections_openclaw(
    body: dict[str, Any],
    *,
    strip_assistant_thinking: bool = True,
) -> list[str]:
    """Extract and serialize assistant response from OpenAI/OpenClaw format.

    Response body is: {"choices": [{"message": {"role": "assistant", "content": "..."}}], ...}
    """
    choices = body.get("choices", [])
    if not choices:
        return []
    choice = choices[0] if isinstance(choices, list) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "")
    if not content:
        return []
    normalized = _normalize_content(
        content,
        strip_assistant_thinking=strip_assistant_thinking,
        role="assistant",
    )
    if not normalized:
        return []
    msg: dict[str, Any] = {"role": "assistant", "content": normalized}
    return [_stable_json({"section": "message", "message": msg})]


def _build_compact_response_body_openclaw(body: dict[str, Any]) -> dict[str, Any]:
    """Build a compact version of the OpenAI/OpenClaw response body."""
    compact: dict[str, Any] = {}

    for key in ("id", "object", "model"):
        if key in body:
            compact[key] = body[key]

    if "usage" in body:
        compact["usage"] = body["usage"]

    if "cache_metrics" in body:
        compact["cache_metrics"] = body["cache_metrics"]

    choices = body.get("choices", [])
    if choices and isinstance(choices, list):
        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, str) and len(content) > 300:
                content = content[:300] + "..."
            compact["choices"] = [{"message": {"role": "assistant", "content": content},
                                   "finish_reason": choice.get("finish_reason")}]

    return compact


# ── Tokenization ───────────────────────────────────────────────────────────

def _get_encoding():
    return tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)


def tokenize_sections(sections: list[str]) -> list[int]:
    if not sections:
        return []
    return list(_get_encoding().encode("\n".join(sections)))


def tokenize_prompt(
    prompt: dict[str, Any],
    *,
    trace_format: str = "claudecode",
) -> tuple[list[str], list[int]]:
    if trace_format == "openclaw":
        sections = _serialize_prompt_sections_openclaw(prompt)
    else:
        sections = _serialize_prompt_sections(prompt)
    return sections, tokenize_sections(sections)


# ── Matching helpers ───────────────────────────────────────────────────────

def _select_non_overlapping_matches(
    matches: list[BlendMatchResult],
    *,
    prefix_token_count: int,
    total_token_count: int,
) -> list[BlendMatchResult]:
    selected: list[BlendMatchResult] = []
    seen_storage_hashes: set[bytes] = set()
    next_free = prefix_token_count

    for match in sorted(matches, key=lambda item: (item.cur_st, item.cur_ed)):
        if match.cur_st < prefix_token_count:
            continue
        if match.cur_ed > total_token_count:
            continue
        if match.cur_st < next_free:
            continue
        if match.storage_hash in seen_storage_hashes:
            continue
        selected.append(match)
        seen_storage_hashes.add(match.storage_hash)
        next_free = match.cur_ed
    return selected


def _extract_session_id(record: dict[str, Any]) -> str:
    headers = record.get("headers") or {}
    return str(
        headers.get("x-claude-code-session-id")
        or headers.get("x-openclaw-session-key")
        or "unknown_session"
    )


# ── Dashboard analysis ─────────────────────────────────────────────────────


def _parse_jsonl_text(trace_text: str) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(trace_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        records.append((line_number, record))
    return records


@dataclass(frozen=True)
class PrefixChunkOrigin:
    source_request_index: int
    first_seen_request_index: int
    from_decode_cache: bool
    storage_hash_id: str


@dataclass
class PrefixSequence:
    request_index: int
    hashes: list[bytes]
    origins: list[PrefixChunkOrigin]


def _best_prefix_match(
    current_hashes: list[bytes],
    previous_sequences: list[PrefixSequence],
) -> tuple[int, PrefixSequence | None]:
    best_count = 0
    best_sequence: PrefixSequence | None = None
    for sequence in previous_sequences:
        limit = min(len(current_hashes), len(sequence.hashes))
        index = 0
        while index < limit and current_hashes[index] == sequence.hashes[index]:
            index += 1
        if index > best_count:
            best_count = index
            best_sequence = sequence
            if best_count == len(current_hashes):
                break
    return best_count, best_sequence


def _preview_text(sections: list[str], *, limit: int = 240) -> str:
    return "\n".join(sections)[:limit]


def _truncate_content_blocks(
    content: Any,
    *,
    text_limit: int = 300,
) -> Any:
    """Truncate text in content blocks for compact display."""
    if isinstance(content, str):
        return content[:text_limit] + ("..." if len(content) > text_limit else "")
    if not isinstance(content, list):
        return content
    result: list[Any] = []
    for block in content:
        if isinstance(block, dict):
            cb = dict(block)
            if "text" in cb and isinstance(cb["text"], str) and len(cb["text"]) > text_limit:
                cb["text"] = cb["text"][:text_limit] + "..."
            # Truncate tool input if present
            if "input" in cb and isinstance(cb["input"], (str, dict)):
                inp = cb["input"]
                if isinstance(inp, str) and len(inp) > text_limit:
                    cb["input"] = inp[:text_limit] + "..."
                elif isinstance(inp, dict):
                    inp_str = json.dumps(inp, ensure_ascii=False)
                    if len(inp_str) > text_limit:
                        cb["input"] = json.loads(inp_str[:text_limit] + "...")  if False else f"({len(inp_str)} chars)"
            result.append(cb)
        else:
            result.append(block)
    return result


def _build_compact_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Build a compact version of the request body for raw trace display.

    Follows ProxyTrace convention: tools stripped to count, system truncated,
    messages stored as count (expanded separately), other fields kept as-is.
    Order: model → tools → system → thinking → other params → messages count.
    """
    compact: dict[str, Any] = {}

    # Model first
    if "model" in body:
        compact["model"] = body["model"]

    # Speed / effort
    if "speed" in body:
        compact["speed"] = body["speed"]
    if "output_config" in body:
        compact["output_config"] = body["output_config"]

    # Tools — replace with count
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        compact["tools"] = f"[{len(tools)} tool definitions stripped]"
    elif isinstance(tools, list):
        compact["tools"] = []

    # Stream
    if "stream" in body:
        compact["stream"] = body["stream"]

    # System — keep structure but truncate text
    raw_sys = body.get("system")
    if raw_sys is not None:
        if isinstance(raw_sys, str):
            compact["system"] = raw_sys[:500] + ("..." if len(raw_sys) > 500 else "")
        elif isinstance(raw_sys, list):
            compact["system"] = _truncate_content_blocks(raw_sys, text_limit=300)
        else:
            compact["system"] = raw_sys

    # Thinking
    if "thinking" in body:
        compact["thinking"] = body["thinking"]

    # Max tokens
    for key in ("max_tokens", "maxTokens"):
        if key in body:
            compact[key] = body[key]

    # Temperature, top_p, top_k
    for key in ("temperature", "top_p", "top_k", "stop_sequences"):
        if key in body:
            compact[key] = body[key]

    # Metadata
    if "metadata" in body:
        compact["metadata"] = body["metadata"]

    # Messages — just the count, actual messages shown separately
    messages = body.get("messages", [])
    compact["messages"] = f"[{len(messages)} messages — see below]"

    return compact


def _build_compact_response_body(body: dict[str, Any]) -> dict[str, Any]:
    """Build a compact version of the response body for raw trace display."""
    compact: dict[str, Any] = {}

    for key in ("type", "id", "model", "role", "stop_reason"):
        if key in body:
            compact[key] = body[key]

    if "usage" in body:
        compact["usage"] = body["usage"]

    content = body.get("content")
    if isinstance(content, list):
        compact["content"] = _truncate_content_blocks(content, text_limit=300)
    elif content is not None:
        compact["content"] = content

    return compact


def _build_visual_chunks(
    *,
    input_tokens: int,
    chunk_size: int,
    prefix_chunk_count: int,
    prefix_sequence: PrefixSequence | None,
    selected_matches: list[BlendMatchResult],
    token_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    prefix_token_count = prefix_chunk_count * chunk_size
    boundaries = {0, input_tokens}

    # Always expose the full prompt on chunk-size boundaries so new-only turns
    # are still trackable in the dashboard. Without this, a request with no
    # prefix/non-prefix matches can collapse into one large untrackable region.
    for token_offset in range(0, input_tokens, chunk_size):
        boundaries.add(token_offset)
        boundaries.add(min(token_offset + chunk_size, input_tokens))

    for chunk_index in range(prefix_chunk_count):
        boundaries.add(chunk_index * chunk_size)
        boundaries.add(min((chunk_index + 1) * chunk_size, input_tokens))

    for match in selected_matches:
        boundaries.add(match.cur_st)
        boundaries.add(match.cur_ed)

    sorted_boundaries = sorted(boundaries)
    chunks: list[dict[str, Any]] = []

    for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
        if end <= start:
            continue

        try:
            text_preview = (
                _get_encoding().decode(list(token_ids[start : min(end, start + 512)]))[:200]
                if token_ids is not None
                else ""
            )
        except Exception:
            text_preview = ""

        # Compute a content-only fingerprint for any exactly-chunk_size slice so
        # every aligned 256-token block is trackable across turns regardless of kind.
        content_fingerprint_id: str | None = None
        if token_ids is not None and (end - start) == chunk_size and end <= len(token_ids):
            try:
                fp = _poly_hash(tuple(token_ids[start:end]))
                content_fingerprint_id = _make_fingerprint_id(fp)
            except Exception:
                pass

        chunk: dict[str, Any] = {
            "chunk_index": len(chunks),
            "token_start": start,
            "token_end": end,
            "token_count": end - start,
            "kind": "new",
            "fingerprint_id": content_fingerprint_id,
            "match_fingerprint_id": None,
            "storage_hash_id": None,
            "source_request_index": None,
            "first_seen_request_index": None,
            "from_decode_cache": False,
            "text_preview": text_preview,
            "text_value": (
                _get_encoding().decode(list(token_ids[start:end]))
                if token_ids is not None
                else ""
            ),
        }

        if end <= prefix_token_count:
            prefix_chunk_index = min(start // chunk_size, prefix_chunk_count - 1)
            origin = (
                prefix_sequence.origins[prefix_chunk_index]
                if prefix_sequence is not None and prefix_chunk_index < len(prefix_sequence.origins)
                else None
            )
            chunk.update(
                {
                    "kind": "prefix",
                    "storage_hash_id": origin.storage_hash_id if origin else None,
                    "source_request_index": (
                        origin.source_request_index if origin else None
                    ),
                    "first_seen_request_index": (
                        origin.first_seen_request_index if origin else None
                    ),
                    "from_decode_cache": origin.from_decode_cache if origin else False,
                }
            )
            chunks.append(chunk)
            continue

        covering_match = next(
            (
                match
                for match in selected_matches
                if match.cur_st <= start and end <= match.cur_ed
            ),
            None,
        )
        if covering_match is not None:
            chunk.update(
                {
                    "kind": "nonprefix",
                    "match_fingerprint_id": covering_match.fingerprint_id,
                    "storage_hash_id": covering_match.storage_hash_id,
                    "source_request_index": covering_match.source_request_index,
                    "first_seen_request_index": covering_match.first_seen_request_index,
                    "from_decode_cache": covering_match.from_decode_cache,
                }
            )

        chunks.append(chunk)

    return chunks


def _overlap_len(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _build_aligned_full_chunks(
    *,
    input_tokens: int,
    chunk_size: int,
    prefix_chunk_count: int,
    prefix_sequence: PrefixSequence | None,
    selected_matches: list[BlendMatchResult],
    token_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    prefix_token_count = prefix_chunk_count * chunk_size
    complete_end = input_tokens - (input_tokens % chunk_size)
    chunks: list[dict[str, Any]] = []

    for start in range(0, complete_end, chunk_size):
        end = start + chunk_size
        try:
            text_preview = (
                _get_encoding().decode(list(token_ids[start : min(end, start + 512)]))[:200]
                if token_ids is not None
                else ""
            )
        except Exception:
            text_preview = ""

        content_fingerprint_id: str | None = None
        if token_ids is not None and end <= len(token_ids):
            try:
                fp = _poly_hash(tuple(token_ids[start:end]))
                content_fingerprint_id = _make_fingerprint_id(fp)
            except Exception:
                pass

        chunk: dict[str, Any] = {
            "chunk_index": len(chunks),
            "token_start": start,
            "token_end": end,
            "token_count": chunk_size,
            "kind": "new",
            "fingerprint_id": content_fingerprint_id,
            "match_fingerprint_id": None,
            "storage_hash_id": None,
            "source_request_index": None,
            "first_seen_request_index": None,
            "from_decode_cache": False,
            "text_preview": text_preview,
            "text_value": (
                _get_encoding().decode(list(token_ids[start:end]))
                if token_ids is not None
                else ""
            ),
        }

        if end <= prefix_token_count:
            prefix_chunk_index = min(start // chunk_size, prefix_chunk_count - 1)
            origin = (
                prefix_sequence.origins[prefix_chunk_index]
                if prefix_sequence is not None and prefix_chunk_index < len(prefix_sequence.origins)
                else None
            )
            chunk.update(
                {
                    "kind": "prefix",
                    "storage_hash_id": origin.storage_hash_id if origin else None,
                    "source_request_index": (
                        origin.source_request_index if origin else None
                    ),
                    "first_seen_request_index": (
                        origin.first_seen_request_index if origin else None
                    ),
                    "from_decode_cache": origin.from_decode_cache if origin else False,
                }
            )
            chunks.append(chunk)
            continue

        best_match: BlendMatchResult | None = None
        best_overlap = 0
        for match in selected_matches:
            overlap = _overlap_len(start, end, match.cur_st, match.cur_ed)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = match

        if best_match is not None and best_overlap * 2 >= chunk_size:
            chunk.update(
                {
                    "kind": "nonprefix",
                    "match_fingerprint_id": best_match.fingerprint_id,
                    "storage_hash_id": best_match.storage_hash_id,
                    "source_request_index": best_match.source_request_index,
                    "first_seen_request_index": best_match.first_seen_request_index,
                    "from_decode_cache": best_match.from_decode_cache,
                }
            )

        chunks.append(chunk)

    return chunks


@dataclass
class RequestMetrics:
    request_index: int
    request_id: str
    raw_line_number: int
    input_tokens_est: int
    response_output_tokens_est: int
    total_tokens_est: int
    prefix_tokens_est: int
    nonprefix_reusable_tokens_est: int
    new_compute_tokens_est: int
    total_complete_chunks: int
    prefix_chunk_count: int
    nonprefix_reusable_chunk_count: int
    chunk_size: int
    prompt_preview: str
    response_preview: str
    prefix_source_request_index: int | None
    chunks: list[dict[str, Any]]
    aligned_full_chunks: list[dict[str, Any]]
    agent_type: str = "main"
    agent_group_id: int = 0
    model: str = ""
    system_preview: str = ""
    tool_names: list[str] = field(default_factory=list)
    messages_summary: list[dict[str, Any]] = field(default_factory=list)
    request_body_compact: dict[str, Any] = field(default_factory=dict)
    response_body_compact: dict[str, Any] = field(default_factory=dict)
    agent_handoffs: list[dict[str, Any]] = field(default_factory=list)
    timestamp_utc: str = ""
    timestamp_rel_s: float | None = None
    response_timestamp_utc: str = ""
    response_timestamp_rel_s: float | None = None
    duration_s: float | None = None
    provider_input_tokens: int | None = None
    provider_cache_read_input_tokens: int | None = None
    provider_cache_creation_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    prompt_sections: list[str] = field(default_factory=list, repr=False)
    prompt_token_ids: list[int] = field(default_factory=list, repr=False)
    prompt_hashes: list[bytes] = field(default_factory=list, repr=False)
    response_sections: list[str] = field(default_factory=list, repr=False)
    response_token_ids: list[int] = field(default_factory=list, repr=False)

    def to_dict(
        self,
        *,
        include_prompt_token_ids: bool = False,
        include_prompt_text: bool = False,
    ) -> dict[str, Any]:
        chunks = [dict(chunk) for chunk in self.chunks]
        aligned_full_chunks = [dict(chunk) for chunk in self.aligned_full_chunks]
        if not include_prompt_text:
            for chunk in chunks:
                chunk.pop("text_value", None)
            for chunk in aligned_full_chunks:
                chunk.pop("text_value", None)
        payload = {
            "request_index": self.request_index,
            "request_id": self.request_id,
            "raw_line_number": self.raw_line_number,
            "agent_type": self.agent_type,
            "agent_group_id": self.agent_group_id,
            "model": self.model,
            "system_preview": self.system_preview,
            "tool_names": self.tool_names,
            "messages_summary": self.messages_summary,
            "request_body_compact": self.request_body_compact,
            "response_body_compact": self.response_body_compact,
            "agent_handoffs": self.agent_handoffs,
            "timestamp_utc": self.timestamp_utc,
            "timestamp_rel_s": self.timestamp_rel_s,
            "response_timestamp_utc": self.response_timestamp_utc,
            "response_timestamp_rel_s": self.response_timestamp_rel_s,
            "duration_s": self.duration_s,
            "input_tokens_est": self.input_tokens_est,
            "response_output_tokens_est": self.response_output_tokens_est,
            "total_tokens_est": self.total_tokens_est,
            "prefix_tokens_est": self.prefix_tokens_est,
            "nonprefix_reusable_tokens_est": self.nonprefix_reusable_tokens_est,
            "new_compute_tokens_est": self.new_compute_tokens_est,
            "prefix_ratio_est": (
                round(self.prefix_tokens_est / self.input_tokens_est, 6)
                if self.input_tokens_est
                else 0.0
            ),
            "nonprefix_reuse_ratio_est": (
                round(self.nonprefix_reusable_tokens_est / self.input_tokens_est, 6)
                if self.input_tokens_est
                else 0.0
            ),
            "new_compute_ratio_est": (
                round(self.new_compute_tokens_est / self.input_tokens_est, 6)
                if self.input_tokens_est
                else 0.0
            ),
            "total_complete_chunks": self.total_complete_chunks,
            "prefix_chunk_count": self.prefix_chunk_count,
            "nonprefix_reusable_chunk_count": self.nonprefix_reusable_chunk_count,
            "prefix_source_request_index": self.prefix_source_request_index,
            "prompt_preview": self.prompt_preview,
            "response_preview": self.response_preview,
            "provider_input_tokens": self.provider_input_tokens,
            "provider_cache_read_input_tokens": self.provider_cache_read_input_tokens,
            "provider_cache_creation_input_tokens": self.provider_cache_creation_input_tokens,
            "provider_output_tokens": self.provider_output_tokens,
            "chunk_size": self.chunk_size,
            "chunks": chunks,
            "aligned_full_chunks": aligned_full_chunks,
            "prompt_sections_count": len(self.prompt_sections),
        }
        if include_prompt_token_ids:
            payload["prompt_token_ids"] = self.prompt_token_ids
        if include_prompt_text:
            payload["prompt_text"] = _get_encoding().decode(list(self.prompt_token_ids))
        return payload


@dataclass
class SessionState:
    session_id: str
    chunk_size: int
    cache_source_mode: CacheSourceMode
    strip_billing_header: bool = True
    trace_format: str = "claudecode"
    prefix_hasher: RollingPrefixHasher = field(init=False)
    matcher: BlendTokenRangeMatcher = field(init=False)
    prefix_sequences: list[PrefixSequence] = field(default_factory=list)
    requests: list[RequestMetrics] = field(default_factory=list)
    count_tokens_request_count: int = 0
    _agent_group_counter: int = field(default=0, init=False)
    _prev_tool_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.prefix_hasher = RollingPrefixHasher(self.chunk_size)
        self.matcher = BlendTokenRangeMatcher(self.chunk_size)

    def process_request(
        self,
        record: dict[str, Any],
        *,
        line_number: int,
        strip_assistant_thinking: bool,
    ) -> RequestMetrics:
        request_index = len(self.requests) + 1
        body = record.get("body") or {}

        # Extract timestamps
        req_timestamp_utc = str(record.get("timestamp_utc", ""))
        req_timestamp_rel_s = record.get("timestamp_rel_s")
        if req_timestamp_rel_s is not None:
            try:
                req_timestamp_rel_s = float(req_timestamp_rel_s)
            except (ValueError, TypeError):
                req_timestamp_rel_s = None

        # Detect agent type and assign group ID
        if self.trace_format == "openclaw":
            agent_type = _detect_agent_type_openclaw(body)
            tool_fp = _compute_tool_fingerprint_openclaw(body)
        else:
            agent_type = _detect_agent_type(body)
            tool_fp = _compute_tool_fingerprint(body)
        if tool_fp != self._prev_tool_fingerprint:
            self._agent_group_counter += 1
            self._prev_tool_fingerprint = tool_fp

        # Extract model name, system preview, tool names, messages summary
        model_name = body.get("model", "")

        if self.trace_format == "openclaw":
            # OpenClaw: system is messages[0] with role="system"
            raw_messages = body.get("messages", [])
            system_preview = ""
            if raw_messages and isinstance(raw_messages[0], dict) and raw_messages[0].get("role") == "system":
                sys_content = raw_messages[0].get("content", "")
                if isinstance(sys_content, str):
                    system_preview = sys_content[:500]
            # OpenAI function format: {"type": "function", "function": {"name": ...}}
            raw_tools = body.get("tools", [])
            tool_names_list = [
                t.get("function", {}).get("name", "")
                for t in raw_tools
                if isinstance(t, dict)
            ]
        else:
            raw_system = body.get("system", [])
            system_preview = ""
            if isinstance(raw_system, str):
                system_preview = raw_system[:500]
            elif isinstance(raw_system, list):
                parts = []
                for sb in raw_system:
                    if isinstance(sb, dict):
                        parts.append(sb.get("text", "")[:300])
                    elif isinstance(sb, str):
                        parts.append(sb[:300])
                system_preview = "\n".join(parts)[:500]
            raw_tools = body.get("tools", [])
            tool_names_list = [
                t.get("name", "") for t in raw_tools if isinstance(t, dict)
            ]
            raw_messages = body.get("messages", [])

        messages_summary: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            content = msg.get("content", "")
            if isinstance(content, str):
                preview = content[:300]
                length = len(content)
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts.append(block.get("text", "")[:200])
                        elif btype == "tool_use":
                            parts.append(f"[tool_use: {block.get('name', '?')}]")
                        elif btype == "tool_result":
                            parts.append("[tool_result]")
                        elif btype == "thinking":
                            parts.append("[thinking]")
                        else:
                            parts.append(f"[{btype}]")
                preview = " ".join(parts)[:300]
                length = sum(
                    len(b.get("text", "")) if isinstance(b, dict) else len(str(b))
                    for b in content
                )
            else:
                preview = str(content)[:300]
                length = len(str(content))
            messages_summary.append({
                "role": role,
                "preview": preview,
                "content_length": length,
            })

        # Build compact request body for raw trace display
        request_body_compact = _build_compact_request_body(body)

        # Extract agent handoff data (Agent tool spawn/return)
        if self.trace_format == "openclaw":
            agent_handoffs: list[dict[str, Any]] = []
        else:
            agent_handoffs = _extract_agent_handoffs(raw_messages)

        if self.trace_format == "openclaw":
            prompt = build_prompt_structure_openclaw(
                body,
                strip_assistant_thinking=strip_assistant_thinking,
            )
        else:
            prompt = build_prompt_structure(
                body,
                strip_assistant_thinking=strip_assistant_thinking,
                strip_billing_header=self.strip_billing_header,
            )
        prompt_sections, token_ids = tokenize_prompt(prompt, trace_format=self.trace_format)
        input_tokens = len(token_ids)

        current_hashes = self.prefix_hasher.compute_chunk_hashes(token_ids)
        prefix_chunk_count, prefix_sequence = _best_prefix_match(
            current_hashes,
            self.prefix_sequences,
        )
        prefix_tokens = prefix_chunk_count * self.chunk_size

        matches = self.matcher.match_sub_sequence(token_ids, scan_start=prefix_tokens)
        selected_matches = _select_non_overlapping_matches(
            matches,
            prefix_token_count=prefix_tokens,
            total_token_count=input_tokens,
        )
        nonprefix_tokens = len(selected_matches) * self.chunk_size
        new_compute_tokens = max(input_tokens - prefix_tokens - nonprefix_tokens, 0)

        metrics = RequestMetrics(
            request_index=request_index,
            request_id=str(record.get("request_id", "")),
            raw_line_number=line_number,
            input_tokens_est=input_tokens,
            response_output_tokens_est=0,
            total_tokens_est=input_tokens,
            prefix_tokens_est=prefix_tokens,
            nonprefix_reusable_tokens_est=nonprefix_tokens,
            new_compute_tokens_est=new_compute_tokens,
            total_complete_chunks=len(current_hashes),
            prefix_chunk_count=prefix_chunk_count,
            nonprefix_reusable_chunk_count=len(selected_matches),
            chunk_size=self.chunk_size,
            prompt_preview=_preview_text(prompt_sections),
            response_preview="",
            prefix_source_request_index=(
                prefix_sequence.request_index if prefix_sequence is not None else None
            ),
            chunks=_build_visual_chunks(
                input_tokens=input_tokens,
                chunk_size=self.chunk_size,
                prefix_chunk_count=prefix_chunk_count,
                prefix_sequence=prefix_sequence,
                selected_matches=selected_matches,
                token_ids=token_ids,
            ),
            aligned_full_chunks=_build_aligned_full_chunks(
                input_tokens=input_tokens,
                chunk_size=self.chunk_size,
                prefix_chunk_count=prefix_chunk_count,
                prefix_sequence=prefix_sequence,
                selected_matches=selected_matches,
                token_ids=token_ids,
            ),
            agent_type=agent_type,
            agent_group_id=self._agent_group_counter,
            model=model_name,
            system_preview=system_preview,
            tool_names=tool_names_list,
            messages_summary=messages_summary,
            request_body_compact=request_body_compact,
            agent_handoffs=agent_handoffs,
            timestamp_utc=req_timestamp_utc,
            timestamp_rel_s=req_timestamp_rel_s,
            prompt_sections=prompt_sections,
            prompt_token_ids=token_ids,
            prompt_hashes=current_hashes,
        )
        self.requests.append(metrics)

        origins = [
            PrefixChunkOrigin(
                source_request_index=request_index,
                first_seen_request_index=request_index,
                from_decode_cache=False,
                storage_hash_id=storage_hash_id(chunk_hash),
            )
            for chunk_hash in current_hashes
        ]
        self.prefix_sequences.append(
            PrefixSequence(
                request_index=request_index,
                hashes=current_hashes,
                origins=origins,
            )
        )
        self.matcher.on_new_token_hashes(
            token_ids,
            current_hashes,
            source_request_index=request_index,
            from_decode_cache=False,
        )
        return metrics

    def process_response(
        self,
        request_metrics: RequestMetrics,
        body: dict[str, Any],
        *,
        strip_assistant_thinking: bool,
    ) -> None:
        if self.trace_format == "openclaw":
            response_sections = serialize_assistant_response_sections_openclaw(
                body,
                strip_assistant_thinking=strip_assistant_thinking,
            )
        else:
            response_sections = serialize_assistant_response_sections(
                body,
                strip_assistant_thinking=strip_assistant_thinking,
            )
        response_token_ids = tokenize_sections(response_sections)
        request_metrics.response_sections = response_sections
        request_metrics.response_token_ids = response_token_ids
        request_metrics.response_output_tokens_est = len(response_token_ids)
        request_metrics.total_tokens_est = (
            request_metrics.input_tokens_est + request_metrics.response_output_tokens_est
        )
        request_metrics.response_preview = _preview_text(response_sections)
        if self.trace_format == "openclaw":
            request_metrics.response_body_compact = _build_compact_response_body_openclaw(body)
        else:
            request_metrics.response_body_compact = _build_compact_response_body(body)

        if (
            self.cache_source_mode != CacheSourceMode.INCLUDE_DECODE_CACHE
            or not response_token_ids
        ):
            return

        combined_token_ids = request_metrics.prompt_token_ids + response_token_ids
        combined_hashes = self.prefix_hasher.compute_chunk_hashes(combined_token_ids)

        prompt_hash_count = len(request_metrics.prompt_hashes)
        combined_origins: list[PrefixChunkOrigin] = [
            PrefixChunkOrigin(
                source_request_index=request_metrics.request_index,
                first_seen_request_index=request_metrics.request_index,
                from_decode_cache=False,
                storage_hash_id=storage_hash_id(chunk_hash),
            )
            for chunk_hash in combined_hashes[:prompt_hash_count]
        ]
        combined_origins.extend(
            PrefixChunkOrigin(
                source_request_index=request_metrics.request_index,
                first_seen_request_index=request_metrics.request_index,
                from_decode_cache=True,
                storage_hash_id=storage_hash_id(chunk_hash),
            )
            for chunk_hash in combined_hashes[prompt_hash_count:]
        )
        self.prefix_sequences.append(
            PrefixSequence(
                request_index=request_metrics.request_index,
                hashes=combined_hashes,
                origins=combined_origins,
            )
        )

        request_complete_tokens = (
            len(request_metrics.prompt_token_ids)
            - (len(request_metrics.prompt_token_ids) % self.chunk_size)
        )
        additional_hashes = self.prefix_hasher.compute_chunk_hashes(
            combined_token_ids,
            start=request_complete_tokens,
        )
        if not additional_hashes:
            return

        suffix_complete_tokens = (
            len(combined_token_ids)
            - (len(combined_token_ids) % self.chunk_size)
            - request_complete_tokens
        )
        if suffix_complete_tokens <= 0:
            return

        suffix_tokens = combined_token_ids[
            request_complete_tokens : request_complete_tokens + suffix_complete_tokens
        ]
        self.matcher.on_new_token_hashes(
            suffix_tokens,
            additional_hashes,
            source_request_index=request_metrics.request_index,
            from_decode_cache=True,
        )

    def to_dict(
        self,
        *,
        include_prompt_token_ids: bool = False,
        include_prompt_text: bool = False,
    ) -> dict[str, Any]:
        input_tokens = sum(request.input_tokens_est for request in self.requests)
        response_output_tokens = sum(
            request.response_output_tokens_est for request in self.requests
        )
        total_tokens = sum(request.total_tokens_est for request in self.requests)
        prefix_tokens = sum(request.prefix_tokens_est for request in self.requests)
        nonprefix_tokens = sum(
            request.nonprefix_reusable_tokens_est for request in self.requests
        )
        new_compute_tokens = sum(
            request.new_compute_tokens_est for request in self.requests
        )
        provider_input_tokens = sum(
            request.provider_input_tokens or 0 for request in self.requests
        )
        provider_cache_read_tokens = sum(
            request.provider_cache_read_input_tokens or 0 for request in self.requests
        )
        provider_cache_creation_tokens = sum(
            request.provider_cache_creation_input_tokens or 0 for request in self.requests
        )
        provider_output_tokens = sum(
            request.provider_output_tokens or 0 for request in self.requests
        )

        # Build agent groups from request sequence
        agent_groups: list[dict[str, Any]] = []
        subagent_counter = 0
        compact_counter = 0
        for request in self.requests:
            if not agent_groups or agent_groups[-1]["group_id"] != request.agent_group_id:
                if request.agent_type == "subagent":
                    subagent_counter += 1
                    label = f"Subagent {subagent_counter}"
                elif request.agent_type == "compact":
                    compact_counter += 1
                    label = f"Compact {compact_counter}"
                elif request.agent_type == "no_tools":
                    label = "No Tools"
                else:
                    label = request.agent_type.capitalize()
                agent_groups.append({
                    "group_id": request.agent_group_id,
                    "agent_type": request.agent_type,
                    "label": label,
                    "request_indices": [],
                })
            agent_groups[-1]["request_indices"].append(request.request_index)

        return {
            "session_id": self.session_id,
            "request_count": len(self.requests),
            "count_tokens_request_count": self.count_tokens_request_count,
            "input_tokens_est": input_tokens,
            "response_output_tokens_est": response_output_tokens,
            "total_tokens_est": total_tokens,
            "prefix_tokens_est": prefix_tokens,
            "nonprefix_reusable_tokens_est": nonprefix_tokens,
            "new_compute_tokens_est": new_compute_tokens,
            "prefix_ratio_est": round(prefix_tokens / input_tokens, 6) if input_tokens else 0.0,
            "nonprefix_reuse_ratio_est": (
                round(nonprefix_tokens / input_tokens, 6) if input_tokens else 0.0
            ),
            "new_compute_ratio_est": (
                round(new_compute_tokens / input_tokens, 6) if input_tokens else 0.0
            ),
            "provider_input_tokens": provider_input_tokens,
            "provider_cache_read_input_tokens": provider_cache_read_tokens,
            "provider_cache_creation_input_tokens": provider_cache_creation_tokens,
            "provider_output_tokens": provider_output_tokens,
            "agent_groups": agent_groups,
            "requests": [
                request.to_dict(
                    include_prompt_token_ids=include_prompt_token_ids,
                    include_prompt_text=include_prompt_text,
                )
                for request in self.requests
            ],
        }


def _summarize_sessions(
    *,
    session_summaries: list[dict[str, Any]],
    trace_label: str,
    chunk_size: int,
    cache_source_mode: CacheSourceMode,
    strip_assistant_thinking: bool,
    skipped_record_count: int,
    trace_format: str = "claudecode",
) -> dict[str, Any]:
    combined_input = sum(item["input_tokens_est"] for item in session_summaries)
    combined_output = sum(item["response_output_tokens_est"] for item in session_summaries)
    combined_total = sum(item["total_tokens_est"] for item in session_summaries)
    combined_prefix = sum(item["prefix_tokens_est"] for item in session_summaries)
    combined_nonprefix = sum(
        item["nonprefix_reusable_tokens_est"] for item in session_summaries
    )
    combined_new_compute = sum(
        item["new_compute_tokens_est"] for item in session_summaries
    )

    return {
        "trace_file": trace_label,
        "analysis_method": "offline_cacheblend_two_hash",
        "tokenizer": "tiktoken-o200k_base",
        "chunk_size": chunk_size,
        "cache_source_mode": cache_source_mode.value,
        "strip_assistant_thinking": strip_assistant_thinking,
        "trace_format": trace_format,
        "notes": (
            [
                "Only type=request records with path=/v1/chat/completions are scored.",
                "Prompt reconstruction uses OpenAI/OpenClaw format: messages (system is messages[0]).",
                "Top-level request metadata like model/max_completion_tokens/stream is excluded from token counts.",
                "Output tokens are counted in total tokens for both cache modes.",
                "Prefix/non-prefix/new-compute ratios are normalized by input tokens.",
            ] if trace_format == "openclaw" else [
                "Only type=request records with path=/v1/messages are scored.",
                "Prompt reconstruction uses Anthropic model-visible fields: tools, system, messages.",
                "Top-level request metadata like model/max_tokens/stream is excluded from token counts.",
                "Previous assistant thinking blocks are stripped by default.",
                "Output tokens are counted in total tokens for both cache modes.",
                "In include_decode_cache mode, assistant responses become future cache sources only after the response arrives.",
                "Prefix/non-prefix/new-compute ratios are normalized by input tokens.",
            ]
        ),
        "session_count": len(session_summaries),
        "skipped_record_count": skipped_record_count,
        "combined": {
            "input_tokens_est": combined_input,
            "response_output_tokens_est": combined_output,
            "total_tokens_est": combined_total,
            "prefix_tokens_est": combined_prefix,
            "nonprefix_reusable_tokens_est": combined_nonprefix,
            "new_compute_tokens_est": combined_new_compute,
            "prefix_ratio_est": (
                round(combined_prefix / combined_input, 6) if combined_input else 0.0
            ),
            "nonprefix_reuse_ratio_est": (
                round(combined_nonprefix / combined_input, 6)
                if combined_input
                else 0.0
            ),
            "new_compute_ratio_est": (
                round(combined_new_compute / combined_input, 6)
                if combined_input
                else 0.0
            ),
        },
        "sessions": session_summaries,
    }


def analyze_records_for_mode(
    records: list[tuple[int, dict[str, Any]]],
    *,
    trace_label: str,
    chunk_size: int,
    strip_assistant_thinking: bool,
    cache_source_mode: CacheSourceMode,
    strip_billing_header: bool = True,
    trace_format: str = "claudecode",
    log_each_entry: bool = False,
    include_prompt_token_ids: bool = False,
    include_prompt_text: bool = False,
) -> dict[str, Any]:
    sessions: dict[str, SessionState] = {}
    inference_request_lookup: dict[str, tuple[str, int]] = {}
    count_tokens_counts: dict[str, int] = {}
    skipped_record_count = 0

    # Path matching depends on trace format
    if trace_format == "openclaw":
        inference_path = "/v1/chat/completions"
    else:
        inference_path = "/v1/messages"

    for line_number, record in records:
        record_type = record.get("type")

        if record_type == "request":
            path_name = record.get("path")
            session_id = _extract_session_id(record)
            if path_name == inference_path:
                session = sessions.setdefault(
                    session_id,
                    SessionState(
                        session_id=session_id,
                        chunk_size=chunk_size,
                        cache_source_mode=cache_source_mode,
                        strip_billing_header=strip_billing_header,
                        trace_format=trace_format,
                    ),
                )
                session.count_tokens_request_count += count_tokens_counts.pop(session_id, 0)
                metrics = session.process_request(
                    record,
                    line_number=line_number,
                    strip_assistant_thinking=strip_assistant_thinking,
                )
                inference_request_lookup[metrics.request_id] = (
                    session_id,
                    len(session.requests) - 1,
                )
                if log_each_entry:
                    print(
                        "[REQ] "
                        f"line={line_number} "
                        f"session={session_id} "
                        f"request={metrics.request_index} "
                        f"input={metrics.input_tokens_est} "
                        f"prefix={metrics.prefix_tokens_est} "
                        f"nonprefix={metrics.nonprefix_reusable_tokens_est} "
                        f"new={metrics.new_compute_tokens_est}",
                        flush=True,
                    )
            elif path_name == "/v1/messages/count_tokens":
                count_tokens_counts[session_id] = count_tokens_counts.get(session_id, 0) + 1
            else:
                skipped_record_count += 1
        elif record_type == "response":
            request_id = str(record.get("request_id", ""))
            lookup = inference_request_lookup.get(request_id)
            if lookup is None:
                continue
            session_id, request_position = lookup
            body = record.get("body")
            if not isinstance(body, dict):
                continue
            request_metrics = sessions[session_id].requests[request_position]
            usage = body.get("usage")
            if isinstance(usage, dict):
                if trace_format == "openclaw":
                    # OpenAI format: prompt_tokens / completion_tokens
                    request_metrics.provider_input_tokens = usage.get("prompt_tokens")
                    request_metrics.provider_output_tokens = usage.get("completion_tokens")
                    # Cache metrics may be in a separate top-level field
                    cache_metrics = body.get("cache_metrics", {})
                    if isinstance(cache_metrics, dict):
                        request_metrics.provider_cache_read_input_tokens = cache_metrics.get("cached_tokens")
                else:
                    # Anthropic format
                    request_metrics.provider_input_tokens = usage.get("input_tokens")
                    request_metrics.provider_cache_read_input_tokens = usage.get(
                        "cache_read_input_tokens"
                    )
                    request_metrics.provider_cache_creation_input_tokens = usage.get(
                        "cache_creation_input_tokens"
                    )
                    request_metrics.provider_output_tokens = usage.get("output_tokens")
            # Extract response timestamps
            resp_ts_utc = str(record.get("timestamp_utc", ""))
            resp_ts_rel = record.get("timestamp_rel_s")
            if resp_ts_rel is not None:
                try:
                    resp_ts_rel = float(resp_ts_rel)
                except (ValueError, TypeError):
                    resp_ts_rel = None
            request_metrics.response_timestamp_utc = resp_ts_utc
            request_metrics.response_timestamp_rel_s = resp_ts_rel
            if request_metrics.timestamp_rel_s is not None and resp_ts_rel is not None:
                request_metrics.duration_s = round(resp_ts_rel - request_metrics.timestamp_rel_s, 2)
            sessions[session_id].process_response(
                request_metrics,
                body,
                strip_assistant_thinking=strip_assistant_thinking,
            )
            if log_each_entry:
                print(
                    "[RESP] "
                    f"line={line_number} "
                    f"session={session_id} "
                    f"request={request_metrics.request_index} "
                    f"output={request_metrics.response_output_tokens_est}",
                    flush=True,
                )
        else:
            skipped_record_count += 1

    session_summaries = [
        session.to_dict(
            include_prompt_token_ids=include_prompt_token_ids,
            include_prompt_text=include_prompt_text,
        )
        for session in sessions.values()
    ]
    return _summarize_sessions(
        session_summaries=session_summaries,
        trace_label=trace_label,
        chunk_size=chunk_size,
        cache_source_mode=cache_source_mode,
        strip_assistant_thinking=strip_assistant_thinking,
        skipped_record_count=skipped_record_count,
        trace_format=trace_format,
    )


def _has_billing_headers(records: list[tuple[int, dict[str, Any]]]) -> bool:
    """Check whether any request in the trace contains a billing header."""
    for _, record in records:
        if record.get("type") != "request":
            continue
        body = record.get("body") or {}
        system = body.get("system")
        if system is None:
            continue
        if isinstance(system, str):
            if system.startswith("x-anthropic-billing-header:"):
                return True
        elif isinstance(system, list):
            for item in system:
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                if text.startswith("x-anthropic-billing-header:"):
                    return True
    return False


def analyze_trace_text(
    trace_text: str,
    *,
    trace_label: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    strip_assistant_thinking: bool = True,
    trace_format: str = "claudecode",
    log_each_entry: bool = False,
    include_prompt_token_ids: bool = False,
    include_prompt_text: bool = False,
) -> dict[str, Any]:
    records = _parse_jsonl_text(trace_text)

    # OpenClaw traces have no billing headers — skip that analysis path
    has_billing = False if trace_format == "openclaw" else _has_billing_headers(records)

    # Primary modes: billing header stripped (best case / CLAUDE_CODE_ATTRIBUTION_HEADER=0)
    mode_summaries = {
        mode.value: analyze_records_for_mode(
            records,
            trace_label=trace_label,
            chunk_size=chunk_size,
            strip_assistant_thinking=strip_assistant_thinking,
            cache_source_mode=mode,
            strip_billing_header=True,
            trace_format=trace_format,
            log_each_entry=log_each_entry,
            include_prompt_token_ids=include_prompt_token_ids,
            include_prompt_text=include_prompt_text,
        )
        for mode in CacheSourceMode
    }

    # Billing-header-included modes (default Claude Code behavior, prefix breaks)
    mode_summaries_with_billing: dict[str, Any] | None = None
    if has_billing:
        mode_summaries_with_billing = {
            mode.value: analyze_records_for_mode(
                records,
                trace_label=trace_label,
                chunk_size=chunk_size,
                strip_assistant_thinking=strip_assistant_thinking,
                cache_source_mode=mode,
                strip_billing_header=False,
                trace_format=trace_format,
                log_each_entry=log_each_entry,
                include_prompt_token_ids=include_prompt_token_ids,
                include_prompt_text=include_prompt_text,
            )
            for mode in CacheSourceMode
        }

    detected_sessions = [
        session["session_id"] for session in mode_summaries[CacheSourceMode.REQUEST_ONLY.value]["sessions"]
    ]
    result: dict[str, Any] = {
        "trace_file": trace_label,
        "analysis_method": "offline_cacheblend_two_hash",
        "tokenizer": "tiktoken-o200k_base",
        "chunk_size": chunk_size,
        "trace_format": trace_format,
        "strip_assistant_thinking": strip_assistant_thinking,
        "has_billing_headers": has_billing,
        "detected_sessions": detected_sessions,
        "modes": mode_summaries,
    }
    if mode_summaries_with_billing is not None:
        result["modes_with_billing_header"] = mode_summaries_with_billing
    return result


def analyze_trace_path(
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    strip_assistant_thinking: bool = True,
    trace_format: str = "claudecode",
    log_each_entry: bool = False,
    include_prompt_token_ids: bool = False,
    include_prompt_text: bool = False,
) -> dict[str, Any]:
    return analyze_trace_text(
        path.read_text(encoding="utf-8"),
        trace_label=str(path.resolve()),
        chunk_size=chunk_size,
        strip_assistant_thinking=strip_assistant_thinking,
        trace_format=trace_format,
        log_each_entry=log_each_entry,
        include_prompt_token_ids=include_prompt_token_ids,
        include_prompt_text=include_prompt_text,
    )


def _strip_for_lightweight_html(analysis: dict[str, Any]) -> dict[str, Any]:
    """Strip heavy fields for a lightweight HTML dashboard.

    Removes raw-trace-tab data (messages_summary, compact bodies, previews),
    and text from chunks — keeps only chunk position/kind/fingerprint metadata
    needed for the visualization.
    """
    import copy
    result = copy.deepcopy(analysis)

    for mode_key in ("modes", "modes_with_billing_header"):
        modes = result.get(mode_key)
        if not isinstance(modes, dict):
            continue
        for mode_data in modes.values():
            for session in mode_data.get("sessions", []):
                for req in session.get("requests", []):
                    # Raw trace tab data
                    req.pop("messages_summary", None)
                    req.pop("request_body_compact", None)
                    req.pop("response_body_compact", None)
                    req.pop("agent_handoffs", None)
                    req.pop("prompt_preview", None)
                    req.pop("response_preview", None)
                    req.pop("system_preview", None)
                    req.pop("tool_names", None)
                    # Timeline-heavy fields are just small numbers, keep them

                    # Strip text from chunks — keep position/kind/fingerprint only
                    for chunk in req.get("chunks", []):
                        chunk.pop("text_preview", None)
                        chunk.pop("text_value", None)
                    for chunk in req.get("aligned_full_chunks", []):
                        chunk.pop("text_preview", None)
                        chunk.pop("text_value", None)

    return result


def generate_html(analysis: dict[str, Any]) -> str:
    """Generate a self-contained HTML dashboard from analysis data."""
    template_path = Path(__file__).resolve().parent / "trace_viewer.html"
    template = template_path.read_text(encoding="utf-8")

    data_json = json.dumps(analysis, ensure_ascii=False)

    return template.replace(
        "/*__TRACE_DATA_PLACEHOLDER__*/null",
        data_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path, help="Raw trace JSONL file.")
    parser.add_argument(
        "--format",
        choices=["claudecode", "openclaw"],
        default="claudecode",
        dest="trace_format",
        help="Trace format: 'claudecode' (Anthropic, default) or 'openclaw' (OpenAI/vLLM).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <stem>_analysis.json next to script.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Generate a self-contained HTML dashboard at this path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Block size in analysis tokens. Default: 256.",
    )
    parser.add_argument(
        "--include-assistant-thinking",
        action="store_true",
        help="Keep prior assistant thinking blocks in the reconstructed prompt.",
    )
    parser.add_argument(
        "--log-each-entry",
        action="store_true",
        help="Print one compact log line for each scored request and response.",
    )
    args = parser.parse_args()

    # For openclaw, never embed full prompt text — sessions are too long
    include_prompt_text = args.html is not None and args.trace_format != "openclaw"

    payload = analyze_trace_path(
        args.trace_path.resolve(),
        chunk_size=args.chunk_size,
        strip_assistant_thinking=not args.include_assistant_thinking,
        trace_format=args.trace_format,
        log_each_entry=args.log_each_entry,
        include_prompt_text=include_prompt_text,
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else Path(__file__).resolve().parent
        / f"{args.trace_path.stem}_analysis.json"
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"JSON: {output_path}")

    if args.html is not None:
        # For openclaw, strip heavy fields to keep HTML small
        html_data = _strip_for_lightweight_html(payload) if args.trace_format == "openclaw" else payload
        html_content = generate_html(html_data)
        html_path = args.html.resolve()
        html_path.write_text(html_content, encoding="utf-8")
        print(f"HTML: {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
