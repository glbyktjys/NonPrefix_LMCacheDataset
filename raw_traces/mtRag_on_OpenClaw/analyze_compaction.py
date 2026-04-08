"""
Analyze OpenClaw MT-RAG backend traces for compaction behavior.

This script treats raw traces as immutable input and emits derived summaries
under raw_traces/mtRag/analysis/.

Token counts are approximate. The traces do not include MiniMax prompt usage,
so this script uses a simple 4-characters-per-token heuristic on normalized
message text. Ratios are therefore the most trustworthy outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proxy.rollingHashes import build_sequence_state, compare_sequence_states


COMPACT_MARKERS = (
    "Create a structured context checkpoint summary",
    "Additional context from /compact",
)
SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:"


@dataclass
class RequestRecord:
    request_id: str
    index: int
    messages: list[dict[str, Any]]
    compact_request: bool


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages[1:]
    return messages


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _message_text(message: dict[str, Any]) -> str:
    return json.dumps(
        {"role": message.get("role"), "content": _content_to_text(message.get("content"))},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _message_token_estimate(message: dict[str, Any]) -> int:
    return max(1, math.ceil(len(_message_text(message)) / 4))


def _request_token_estimate(messages: list[dict[str, Any]]) -> int:
    return sum(_message_token_estimate(message) for message in messages)


def _is_compact_request(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        text = _content_to_text(message.get("content"))
        if any(marker in text for marker in COMPACT_MARKERS):
            return True
    return False


def _has_inserted_summary(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    return _content_to_text(messages[0].get("content")).startswith(SUMMARY_PREFIX)


def _longest_suffix_reused_in_current(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> tuple[int, int | None]:
    prev_hashes = [_message_text(message) for message in previous]
    curr_hashes = [_message_text(message) for message in current]

    max_len = min(len(prev_hashes), len(curr_hashes))
    for length in range(max_len, 0, -1):
        suffix = prev_hashes[-length:]
        for start in range(len(curr_hashes) - length + 1):
            if curr_hashes[start:start + length] == suffix:
                return length, start
    return 0, None


def _tail_token_estimate(messages: list[dict[str, Any]], tail_len: int) -> int:
    if tail_len <= 0:
        return 0
    return _request_token_estimate(messages[-tail_len:])


def _load_requests(path: Path) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    with path.open() as handle:
        request_index = 0
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "request" or row.get("path") != "/v1/chat/completions":
                continue
            request_index += 1
            body = row.get("body") or {}
            messages = _normalize_messages(body.get("messages") or [])
            records.append(
                RequestRecord(
                    request_id=str(row.get("request_id")),
                    index=request_index,
                    messages=messages,
                    compact_request=_is_compact_request(messages),
                )
            )
    return records


def analyze_trace(path: Path) -> dict[str, Any]:
    requests = _load_requests(path)
    compact_requests = sum(1 for request in requests if request.compact_request)
    bursts: list[dict[str, Any]] = []

    i = 0
    while i < len(requests):
        if not requests[i].compact_request:
            i += 1
            continue

        burst_start = i
        while i + 1 < len(requests) and requests[i + 1].compact_request:
            i += 1
        burst_end = i

        previous_normal = next(
            (request for request in reversed(requests[:burst_start]) if not request.compact_request),
            None,
        )
        next_normal = next(
            (request for request in requests[burst_end + 1:] if not request.compact_request),
            None,
        )

        burst_record: dict[str, Any] = {
            "burst_request_indexes": [request.index for request in requests[burst_start:burst_end + 1]],
            "burst_request_ids": [request.request_id for request in requests[burst_start:burst_end + 1]],
            "burst_length": burst_end - burst_start + 1,
            "previous_normal_request_index": previous_normal.index if previous_normal else None,
            "previous_normal_request_id": previous_normal.request_id if previous_normal else None,
            "next_normal_request_index": next_normal.index if next_normal else None,
            "next_normal_request_id": next_normal.request_id if next_normal else None,
        }

        if previous_normal and next_normal:
            previous_state = build_sequence_state(previous_normal.messages)
            next_state = build_sequence_state(next_normal.messages)
            comparison = compare_sequence_states(next_state, previous_state)
            retained_tail_messages, retained_tail_start = _longest_suffix_reused_in_current(
                previous_normal.messages,
                next_normal.messages,
            )
            total_next_tokens = _request_token_estimate(next_normal.messages)
            summary_tokens = (
                _message_token_estimate(next_normal.messages[0])
                if _has_inserted_summary(next_normal.messages)
                else 0
            )
            retained_tail_tokens = _tail_token_estimate(previous_normal.messages, retained_tail_messages)

            burst_record.update(
                {
                    "previous_normal_message_count": len(previous_normal.messages),
                    "next_normal_message_count": len(next_normal.messages),
                    "next_has_inserted_summary": _has_inserted_summary(next_normal.messages),
                    "next_total_approx_tokens": total_next_tokens,
                    "next_summary_approx_tokens": summary_tokens,
                    "direct_prefix_chunk_ratio": comparison["shared_prefix_chunk_ratio"],
                    "direct_prefix_message_ratio": comparison["shared_prefix_message_ratio"],
                    "reused_nonprefix_chunk_ratio": comparison["reused_nonprefix_chunk_ratio"],
                    "reused_nonprefix_chunk_count": comparison["reused_nonprefix_chunk_count"],
                    "retained_tail_message_count": retained_tail_messages,
                    "retained_tail_start_index": retained_tail_start,
                    "retained_tail_approx_tokens": retained_tail_tokens,
                    "retained_tail_ratio": round(
                        retained_tail_tokens / total_next_tokens,
                        6,
                    ) if total_next_tokens else 0.0,
                    "summary_ratio": round(
                        summary_tokens / total_next_tokens,
                        6,
                    ) if total_next_tokens else 0.0,
                }
            )

            burst_record["pattern"] = (
                "summary_plus_tail"
                if burst_record["next_has_inserted_summary"] and retained_tail_messages >= 2
                else "summary_without_clear_tail"
                if burst_record["next_has_inserted_summary"]
                else "reset_or_other"
            )
            burst_record["matches_expected_summary_plus_recent_tail"] = (
                burst_record["next_has_inserted_summary"]
                and comparison["shared_prefix_chunk_ratio"] == 0.0
                and retained_tail_messages >= 2
                and comparison["reused_nonprefix_chunk_count"] > 0
            )

        bursts.append(burst_record)
        i += 1

    matched = sum(
        1 for burst in bursts if burst.get("matches_expected_summary_plus_recent_tail")
    )
    summary = {
        "trace_file": str(path),
        "request_count": len(requests),
        "compact_request_count": compact_requests,
        "compact_request_ratio": round(compact_requests / len(requests), 6) if requests else 0.0,
        "compaction_burst_count": len(bursts),
        "bursts_matching_expected_summary_plus_recent_tail": matched,
        "bursts": bursts,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=sorted((Path(__file__).resolve().parent / "traces").glob("*.jsonl")),
        help="Trace JSONL files to analyze. Defaults to raw_traces/mtRag/traces/*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis",
        help="Directory where analysis JSON files should be written.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.paths:
        summary = analyze_trace(path.resolve())
        output_path = args.output_dir / f"{path.stem}_compaction_summary.json"
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"{path.name} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
