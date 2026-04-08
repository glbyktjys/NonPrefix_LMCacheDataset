"""
Utilities for prompt-structure logging and heuristic conversation tracking.

The proxy only sees serialized request payloads, not model tokenizer state, so
the "chunk" metrics here are based on normalized message text split into fixed
character chunks. This keeps the metrics stable and cheap to compute while
still surfacing prefix reuse, non-prefix reuse, and compaction-like rewrites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_CHUNK_SIZE_CHARS = 256
MAX_MATCH_EXAMPLES = 8


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _message_text(message: dict[str, Any]) -> str:
    role = message.get("role", "")
    return _stable_json({"role": role, "message": message})


def _chunk_text(text: str, chunk_size_chars: int) -> list[str]:
    if not text:
        return [""]
    return [
        text[offset: offset + chunk_size_chars]
        for offset in range(0, len(text), chunk_size_chars)
    ]


def _longest_common_prefix(left: list[str], right: list[str]) -> int:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    return index


@dataclass
class SequenceState:
    message_hashes: list[str]
    chunk_hashes: list[str]
    chunk_refs: list[dict[str, int]]
    message_lengths_chars: list[int]
    total_message_chars: int
    total_chunks: int


def build_sequence_state(
    messages: list[dict[str, Any]],
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
) -> SequenceState:
    message_hashes: list[str] = []
    chunk_hashes: list[str] = []
    chunk_refs: list[dict[str, int]] = []
    message_lengths_chars: list[int] = []

    for message_index, message in enumerate(messages):
        text = _message_text(message)
        message_hashes.append(_hash_text(text))
        message_lengths_chars.append(len(text))

        for chunk_index, chunk in enumerate(_chunk_text(text, chunk_size_chars)):
            chunk_hashes.append(_hash_text(chunk))
            chunk_refs.append(
                {
                    "message_index": message_index,
                    "chunk_index": chunk_index,
                }
            )

    return SequenceState(
        message_hashes=message_hashes,
        chunk_hashes=chunk_hashes,
        chunk_refs=chunk_refs,
        message_lengths_chars=message_lengths_chars,
        total_message_chars=sum(message_lengths_chars),
        total_chunks=len(chunk_hashes),
    )


def compare_sequence_states(
    current: SequenceState,
    previous: Optional[SequenceState],
) -> dict[str, Any]:
    if previous is None:
        return {
            "shared_prefix_message_count": 0,
            "shared_prefix_chunk_count": 0,
            "shared_prefix_message_ratio": 0.0,
            "shared_prefix_chunk_ratio": 0.0,
            "reused_nonprefix_chunk_count": 0,
            "reused_nonprefix_chunk_ratio": 0.0,
            "reused_nonprefix_matches": [],
            "relationship": "new_conversation",
            "compaction_flag": False,
            "compaction_reason": None,
        }

    prefix_messages = _longest_common_prefix(
        current.message_hashes,
        previous.message_hashes,
    )
    prefix_chunks = _longest_common_prefix(
        current.chunk_hashes,
        previous.chunk_hashes,
    )

    previous_chunk_first_seen: dict[str, tuple[int, dict[str, int]]] = {}
    for flat_index, (chunk_hash, ref) in enumerate(zip(previous.chunk_hashes, previous.chunk_refs)):
        previous_chunk_first_seen.setdefault(chunk_hash, (flat_index, ref))

    reused_matches: list[dict[str, int]] = []
    reused_nonprefix_count = 0
    for flat_index in range(prefix_chunks, len(current.chunk_hashes)):
        chunk_hash = current.chunk_hashes[flat_index]
        previous_match = previous_chunk_first_seen.get(chunk_hash)
        if previous_match is None:
            continue
        reused_nonprefix_count += 1
        if len(reused_matches) < MAX_MATCH_EXAMPLES:
            previous_flat_index, previous_ref = previous_match
            current_ref = current.chunk_refs[flat_index]
            reused_matches.append(
                {
                    "current_flat_chunk_index": flat_index,
                    "current_message_index": current_ref["message_index"],
                    "current_chunk_index": current_ref["chunk_index"],
                    "previous_flat_chunk_index": previous_flat_index,
                    "previous_message_index": previous_ref["message_index"],
                    "previous_chunk_index": previous_ref["chunk_index"],
                }
            )

    relationship = "diverged"
    if current.message_hashes == previous.message_hashes:
        relationship = "identical"
    elif prefix_messages == len(previous.message_hashes) and len(current.message_hashes) >= len(previous.message_hashes):
        relationship = "append"
    elif prefix_messages == len(current.message_hashes) and len(current.message_hashes) < len(previous.message_hashes):
        relationship = "prefix_truncate"
    elif prefix_messages >= 3 and reused_nonprefix_count > 0:
        relationship = "rewrite"

    compaction_flag = False
    compaction_reason = None
    if relationship == "prefix_truncate":
        compaction_flag = True
        compaction_reason = "message_prefix_truncated"
    elif relationship == "rewrite":
        shrank = (
            len(current.message_hashes) < len(previous.message_hashes)
            or current.total_message_chars < previous.total_message_chars
        )
        if shrank:
            compaction_flag = True
            compaction_reason = "rewrite_with_reused_nonprefix_chunks"

    return {
        "shared_prefix_message_count": prefix_messages,
        "shared_prefix_chunk_count": prefix_chunks,
        "shared_prefix_message_ratio": round(
            prefix_messages / len(current.message_hashes),
            6,
        ) if current.message_hashes else 0.0,
        "shared_prefix_chunk_ratio": round(
            prefix_chunks / len(current.chunk_hashes),
            6,
        ) if current.chunk_hashes else 0.0,
        "reused_nonprefix_chunk_count": reused_nonprefix_count,
        "reused_nonprefix_chunk_ratio": round(
            reused_nonprefix_count / max(len(current.chunk_hashes) - prefix_chunks, 1),
            6,
        ) if current.chunk_hashes and len(current.chunk_hashes) > prefix_chunks else 0.0,
        "reused_nonprefix_matches": reused_matches,
        "relationship": relationship,
        "compaction_flag": compaction_flag,
        "compaction_reason": compaction_reason,
    }


@dataclass
class ConversationSnapshot:
    conversation_id: str
    turn_index: int
    request_id: str
    sequence: SequenceState


class ConversationTracker:
    """
    Heuristically map requests onto conversation-local lineages.

    Because OpenClaw can interleave multiple sessions in one raw trace and the
    upstream request body does not expose a stable session id, we pick the prior
    lineage whose latest request looks most like the current request. The proxy
    stores both the chosen lineage id and the specific prior request id that the
    metrics were computed against, so downstream analysis can audit the match.
    """

    def __init__(self, chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS):
        self.chunk_size_chars = chunk_size_chars
        self._snapshots: list[ConversationSnapshot] = []
        self._latest_turn_by_conversation: dict[str, int] = {}
        self._next_conversation_index = 1

    def reset(self) -> None:
        self._snapshots = []
        self._latest_turn_by_conversation = {}
        self._next_conversation_index = 1

    def analyze_request(
        self,
        request_id: str,
        body: Any,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(body, dict):
            return None

        messages = body.get("messages")
        if not isinstance(messages, list):
            return None

        sequence = build_sequence_state(messages, self.chunk_size_chars)

        best_snapshot: Optional[ConversationSnapshot] = None
        best_comparison: Optional[dict[str, Any]] = None
        best_score: Optional[tuple[int, int, int, int]] = None

        for snapshot in self._snapshots:
            comparison = compare_sequence_states(sequence, snapshot.sequence)
            relationship = comparison["relationship"]
            if relationship not in ("append", "prefix_truncate", "rewrite", "identical"):
                continue

            score = (
                1 if relationship in ("append", "identical") else 0,
                comparison["shared_prefix_message_count"],
                comparison["shared_prefix_chunk_count"],
                comparison["reused_nonprefix_chunk_count"],
            )
            if best_score is None or score > best_score:
                best_snapshot = snapshot
                best_comparison = comparison
                best_score = score

        if best_snapshot is None:
            conversation_id = f"conv_{self._next_conversation_index:04d}"
            self._next_conversation_index += 1
            comparison = compare_sequence_states(sequence, None)
            turn_index = 0
            compared_to_request_id = None
        else:
            conversation_id = best_snapshot.conversation_id
            comparison = best_comparison or compare_sequence_states(sequence, best_snapshot.sequence)
            turn_index = self._latest_turn_by_conversation[conversation_id] + 1
            compared_to_request_id = best_snapshot.request_id

        snapshot = ConversationSnapshot(
            conversation_id=conversation_id,
            turn_index=turn_index,
            request_id=request_id,
            sequence=sequence,
        )
        self._snapshots.append(snapshot)
        self._latest_turn_by_conversation[conversation_id] = turn_index

        return {
            "conversation_id": conversation_id,
            "conversation_turn_index": turn_index,
            "compared_to_request_id": compared_to_request_id,
            "message_count": len(messages),
            "message_lengths_chars": sequence.message_lengths_chars,
            "total_message_chars": sequence.total_message_chars,
            "chunk_size_chars": self.chunk_size_chars,
            "total_chunks": sequence.total_chunks,
            **comparison,
        }
