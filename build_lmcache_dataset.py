"""
Convert raw proxy trace JSONL files into the LMCache dataset format.

Each output row represents one LLM iteration (request/response pair):
  session_id     - unique session identifier
  model          - model name used
  input          - full cumulative OpenAI-format messages array
  pre_gap        - seconds between previous response completing and this request
  output_length  - completion tokens generated
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


JsonDict = Dict[str, Any]


def strip_cache_control(obj: Any) -> Any:
    """Recursively remove Anthropic prompt-caching markers."""
    if isinstance(obj, dict):
        return {
            key: strip_cache_control(value)
            for key, value in obj.items()
            if key != "cache_control"
        }
    if isinstance(obj, list):
        return [strip_cache_control(item) for item in obj]
    return obj


def session_id_from_path(trace_path: Path) -> str:
    """
    Derive a session_id from the trace filename.
    E.g. 'testing_session_trace.jsonl' -> 'testing_session__claude'
    Override by passing an explicit session_id to process_trace().
    """
    stem = trace_path.stem
    name = re.sub(r"_trace$", "", stem)
    return "%s__claude" % name


def classify_thread_type(body: JsonDict) -> str:
    """
    Classify the request into a coarse thread type.

    This is heuristic by design. We keep the original request payload intact,
    but add lightweight lineage metadata so multi-agent traces are easier to
    analyze and replay.
    """
    system_text = block_content_to_text(strip_cache_control(body.get("system"))).lower()

    if "generate a concise, sentence-case title" in system_text:
        return "title"
    if "helpful ai assistant tasked with summarizing conversations" in system_text:
        return "summarizer"
    if "file search specialist" in system_text or "read-only exploration task" in system_text:
        return "subagent"
    return "main"


def block_content_to_text(content: Any) -> str:
    """Flatten Anthropic content blocks into a string payload."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(strip_cache_control(content), ensure_ascii=False)

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue

        if not isinstance(block, dict):
            parts.append(str(block))
            continue

        block = strip_cache_control(block)
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
            continue

        if block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                parts.append(thinking)
            continue

        if block_type == "tool_result":
            tool_content = block.get("content", "")
            parts.append(block_content_to_text(tool_content))
            continue

        parts.append(json.dumps(block, ensure_ascii=False))

    return "\n".join(part for part in parts if part)


def convert_system_blocks(system_blocks: Any) -> List[JsonDict]:
    """Convert Anthropic top-level system blocks into a single system message."""
    if not system_blocks:
        return []

    system_text = block_content_to_text(strip_cache_control(system_blocks))
    if not system_text:
        return []

    return [{"role": "system", "content": system_text}]


def assistant_message_from_blocks(message: JsonDict) -> List[JsonDict]:
    """Convert an Anthropic assistant message into OpenAI-style assistant messages."""
    content = strip_cache_control(message.get("content"))

    if isinstance(content, str):
        return [{"role": "assistant", "content": content}]

    if not isinstance(content, list):
        return [{"role": "assistant", "content": block_content_to_text(content)}]

    text_parts: List[str] = []
    tool_calls: List[JsonDict] = []

    for block in content:
        if not isinstance(block, dict):
            if block:
                text_parts.append(str(block))
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
            continue

        if block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                text_parts.append(thinking)
            continue

        if block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(
                        block.get("input", {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            })
            continue

        text_parts.append(json.dumps(block, ensure_ascii=False))

    assistant_message: JsonDict = {
        "role": "assistant",
        "content": "\n".join(part for part in text_parts if part),
    }
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    return [assistant_message]


def user_message_from_blocks(message: JsonDict) -> List[JsonDict]:
    """
    Convert an Anthropic user message into OpenAI-style user/tool messages.

    Anthropic encodes tool results as blocks inside a user message. We split
    them into separate tool-role messages while preserving order.
    """
    content = strip_cache_control(message.get("content"))

    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    if not isinstance(content, list):
        return [{"role": "user", "content": block_content_to_text(content)}]

    converted: List[JsonDict] = []
    text_parts: List[str] = []

    def flush_user_text() -> None:
        text = "\n".join(part for part in text_parts if part)
        if text:
            converted.append({"role": "user", "content": text})
        text_parts[:] = []

    for block in content:
        if not isinstance(block, dict):
            if block:
                text_parts.append(str(block))
            continue

        block_type = block.get("type")

        if block_type == "tool_result":
            flush_user_text()
            tool_text = block_content_to_text(block.get("content"))
            tool_message: JsonDict = {
                "role": "tool",
                "content": tool_text,
                "tool_call_id": block.get("tool_use_id", ""),
            }
            if block.get("name"):
                tool_message["name"] = block.get("name")
            converted.append(tool_message)
            continue

        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
            continue

        if block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                text_parts.append(thinking)
            continue

        text_parts.append(json.dumps(block, ensure_ascii=False))

    flush_user_text()
    return converted


def convert_message(message: JsonDict) -> List[JsonDict]:
    """Convert a single Anthropic chat message to one or more OpenAI-style messages."""
    role = message.get("role")
    if role == "assistant":
        return assistant_message_from_blocks(message)
    if role == "user":
        return user_message_from_blocks(message)
    return [{"role": role, "content": block_content_to_text(message.get("content"))}]


def convert_request_to_input(body: JsonDict) -> List[JsonDict]:
    """Build the cumulative OpenAI-style input array for one request."""
    converted: List[JsonDict] = []
    converted.extend(convert_system_blocks(body.get("system")))

    for message in body.get("messages", []):
        converted.extend(convert_message(message))

    return converted


def process_trace(
    trace_path: Union[str, Path],
    session_id: Optional[str] = None,
    model_filter: Optional[str] = None,
) -> List[JsonDict]:
    """
    Read a raw proxy trace JSONL and return a list of dataset rows.

    Parameters
    ----------
    trace_path   : path to the raw *_trace.jsonl file
    session_id   : override the auto-derived session_id
    model_filter : if set, only keep iterations whose model contains this
                   substring (e.g. 'sonnet')

    Returns
    -------
    List of dicts with keys: session_id, model, input, pre_gap, output_length
    """
    trace_path = Path(trace_path)

    entries: List[JsonDict] = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    requests: Dict[str, JsonDict] = {}
    responses: Dict[str, JsonDict] = {}

    for entry in entries:
        request_id = entry.get("request_id")
        if not request_id:
            continue

        if entry.get("type") == "request":
            if entry.get("method") != "POST":
                continue
            if entry.get("path") != "/v1/messages":
                continue
            if not isinstance(entry.get("body"), dict):
                continue
            requests[request_id] = entry
        elif entry.get("type") == "response":
            responses[request_id] = entry

    sid = session_id or session_id_from_path(trace_path)

    pairs = []
    for request_id, request in requests.items():
        response = responses.get(request_id)
        pairs.append((request["timestamp_rel_s"], request_id, request, response))
    pairs.sort(key=lambda item: item[0])

    rows: List[JsonDict] = []
    prev_response_end_time: Optional[float] = None
    row_index = 0
    current_thread_ids: Dict[str, str] = {}
    thread_counters: Dict[str, int] = {}
    last_main_request_id: Optional[str] = None

    for req_time, raw_request_id, request, response in pairs:
        row_index += 1
        body = request["body"]
        model = body.get("model", "")
        thread_type = classify_thread_type(body)

        if model_filter and model_filter not in model:
            continue

        if prev_response_end_time is None:
            pre_gap = 0.0
        else:
            pre_gap = max(0.0, req_time - prev_response_end_time)

        output_length = None
        if response is not None and isinstance(response.get("body"), dict):
            usage = response["body"].get("usage", {})
            if isinstance(usage, dict):
                output_length = usage.get("output_tokens")
            prev_response_end_time = response.get("timestamp_rel_s", req_time)
        else:
            prev_response_end_time = req_time

        if thread_type not in current_thread_ids:
            thread_counters[thread_type] = thread_counters.get(thread_type, 0) + 1
            current_thread_ids[thread_type] = "%s_%d" % (
                thread_type,
                thread_counters[thread_type],
            )

        request_id = "req_%04d" % row_index

        if thread_type == "main":
            parent_request_id = None
            last_main_request_id = request_id
        else:
            parent_request_id = last_main_request_id

        rows.append({
            "session_id": sid,
            "request_id": request_id,
            "parent_request_id": parent_request_id,
            "thread_id": current_thread_ids[thread_type],
            "thread_type": thread_type,
            "model": model,
            "input": convert_request_to_input(body),
            "pre_gap": round(pre_gap, 4),
            "output_length": output_length,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw proxy traces to LMCache dataset JSONL."
    )
    parser.add_argument("traces", nargs="+", help="Input *_trace.jsonl file(s)")
    parser.add_argument(
        "-o",
        "--output",
        default="train.jsonl",
        help="Output JSONL path (default: train.jsonl)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Override session_id (only valid for a single input file)",
    )
    parser.add_argument(
        "--model-filter",
        default=None,
        help="Keep only iterations whose model contains this string",
    )
    args = parser.parse_args()

    if args.session_id and len(args.traces) > 1:
        print("--session-id can only be used with a single input file", file=sys.stderr)
        sys.exit(1)

    all_rows: List[JsonDict] = []
    for path in args.traces:
        rows = process_trace(
            path,
            session_id=args.session_id,
            model_filter=args.model_filter,
        )
        all_rows.extend(rows)
        print("%s: %d iterations" % (path, len(rows)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nWrote %d rows -> %s" % (len(all_rows), args.output))


if __name__ == "__main__":
    main()
