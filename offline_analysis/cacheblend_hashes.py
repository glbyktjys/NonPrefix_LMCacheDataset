"""
Offline adaptation of CacheBlend's two-hash design.

This module mirrors the key idea from LMCache CacheBlend:

1. Rolling prefix storage hash
   - position-dependent
   - each chunk hash depends on all prior chunks
   - used as the storage key

2. Polynomial chunk fingerprint
   - content-only and position-independent
   - used to find reusable chunks anywhere in a later request

This is intentionally lightweight and self-contained for offline trace analysis.
It does not depend on vLLM or LMCache runtime internals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Dict, Iterable, List, Sequence


DEFAULT_CHUNK_SIZE = 256
DEFAULT_POLY_BASE = 0x9E3779B97F4A7C15
UINT64_MASK = (1 << 64) - 1

# convert list of tokens to raw bytes
def _pack_tokens(tokens: Sequence[int]) -> bytes:
    return b"".join(struct.pack(">I", int(token) & 0xFFFFFFFF) for token in tokens)

# return fingerprint for a chunk of tokens
def _poly_hash(tokens: Sequence[int], base: int = DEFAULT_POLY_BASE) -> int:
    value = 0
    for token in tokens:
        value = ((value * base) + (int(token) & UINT64_MASK)) & UINT64_MASK
    return value


def chunk_fingerprint_id(fingerprint: int) -> str:
    return f"fp-{fingerprint:016x}"


def storage_hash_id(storage_hash: bytes) -> str:
    return storage_hash.hex()[:16]


# compute fingerprint at every possible starting position given a sequence of tokens id
# search for matches
def rolling_window_fingerprints(
    token_ids: Sequence[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    base: int = DEFAULT_POLY_BASE,
) -> List[int]:
    """Return polynomial rolling hashes for every chunk-sized window."""
    if chunk_size <= 0 or len(token_ids) < chunk_size:
        return []

    power = 1
    for _ in range(chunk_size - 1):
        power = (power * base) & UINT64_MASK

    first_window = token_ids[:chunk_size]
    current = _poly_hash(first_window, base)
    fingerprints = [current]

    for index in range(chunk_size, len(token_ids)):
        old_token = int(token_ids[index - chunk_size]) & UINT64_MASK
        new_token = int(token_ids[index]) & UINT64_MASK
        current = (current - ((old_token * power) & UINT64_MASK)) & UINT64_MASK
        current = ((current * base) + new_token) & UINT64_MASK
        fingerprints.append(current)
    return fingerprints

# insert into cache
def chunk_fingerprints(
    token_ids: Sequence[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    base: int = DEFAULT_POLY_BASE,
) -> List[int]:
    """Return polynomial hashes for non-overlapping complete chunks."""
    chunk_count = len(token_ids) // chunk_size
    return [
        _poly_hash(
            token_ids[index * chunk_size : (index + 1) * chunk_size],
            base,
        )
        for index in range(chunk_count)
    ]


class RollingPrefixHasher:
    """Compute LMCache-style rolling storage hashes for complete chunks."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.none_hash = bytes(32)

    def hash_chunk(
        self,
        tokens: Sequence[int],
        prefix_hash: bytes | None = None,
    ) -> bytes:
        digest = hashlib.blake2b(digest_size=32)
        digest.update(self.none_hash if prefix_hash is None else prefix_hash)
        digest.update(_pack_tokens(tokens))
        return digest.digest()

    def compute_chunk_hashes(
        self,
        token_ids: Sequence[int],
        prefix_hash: bytes | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> List[bytes]:
        """Compute rolling hashes for chunk-aligned ranges."""
        hashes: List[bytes] = []
        prefix_hash = self.none_hash if prefix_hash is None else prefix_hash
        effective_end = min(len(token_ids), end) if end is not None else len(token_ids)
        complete_end = effective_end - (effective_end % self.chunk_size)

        for offset in range(0, complete_end, self.chunk_size):
            prefix_hash = self.hash_chunk(
                token_ids[offset : offset + self.chunk_size],
                prefix_hash,
            )
            if offset >= start:
                hashes.append(prefix_hash)
        return hashes


@dataclass(frozen=True)
class BlendMatchResult:
    old_st: int
    old_ed: int
    cur_st: int
    cur_ed: int
    compact_id: int
    fingerprint: int
    fingerprint_id: str
    storage_hash: bytes
    storage_hash_id: str
    source_request_index: int
    first_seen_request_index: int
    from_decode_cache: bool


class BlendTokenRangeMatcher:
    """Offline matcher for CacheBlend-style non-prefix chunk reuse."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        base: int = DEFAULT_POLY_BASE,
    ):
        self.chunk_size = chunk_size
        self.base = base
        self._fingerprint_to_chunk_ids: Dict[int, List[int]] = {}
        self._chunk_storage_hashes: List[bytes | None] = []
        self._chunk_tokens: List[tuple[int, ...] | None] = []
        self._chunk_fingerprints: List[int | None] = []
        self._chunk_source_request_indexes: List[int | None] = []
        self._chunk_first_seen_request_indexes: List[int | None] = []
        self._chunk_from_decode_cache: List[bool | None] = []
        self._storage_hash_to_start: Dict[bytes, int] = {}
        self._storage_hash_to_compact_id: Dict[bytes, int] = {}
        self._fingerprint_first_seen_request_index: Dict[int, int] = {}

    def on_new_token_hashes(
        self,
        token_ids: Sequence[int],
        storage_hashes: Sequence[bytes],
        *,
        source_request_index: int = 0,
        from_decode_cache: bool = False,
    ) -> None:
        chunk_count = min(len(token_ids) // self.chunk_size, len(storage_hashes))
        if chunk_count == 0:
            return

        for chunk_index in range(chunk_count):
            start = chunk_index * self.chunk_size
            end = start + self.chunk_size
            chunk_tokens = tuple(int(token) for token in token_ids[start:end])
            fingerprint = _poly_hash(chunk_tokens, self.base)
            storage_hash = storage_hashes[chunk_index]
            first_seen_request_index = self._fingerprint_first_seen_request_index.setdefault(
                fingerprint,
                source_request_index,
            )

            compact_id = len(self._chunk_storage_hashes)
            self._chunk_storage_hashes.append(storage_hash)
            self._chunk_tokens.append(chunk_tokens)
            self._chunk_fingerprints.append(fingerprint)
            self._chunk_source_request_indexes.append(source_request_index)
            self._chunk_first_seen_request_indexes.append(first_seen_request_index)
            self._chunk_from_decode_cache.append(from_decode_cache)
            self._storage_hash_to_start[storage_hash] = start
            self._storage_hash_to_compact_id[storage_hash] = compact_id
            self._fingerprint_to_chunk_ids.setdefault(fingerprint, []).append(compact_id)

    def match_sub_sequence(
        self,
        token_ids: Sequence[int],
        scan_start: int = 0,
    ) -> List[BlendMatchResult]:
        """Find cached chunks that match content in token_ids.

        Args:
            token_ids: Full token sequence to scan.
            scan_start: Start scanning from this token offset.  When the
                caller already knows the prefix length, passing it here
                avoids scanning the prefix region.  This prevents the
                ``seen_hashes`` dedup from consuming a stored chunk on a
                prefix-region hit and then blocking the same chunk's
                legitimate match in the post-prefix region.
        """
        scan_tokens = token_ids[scan_start:]
        if len(scan_tokens) < self.chunk_size or not self._chunk_storage_hashes:
            return []

        rolling = rolling_window_fingerprints(scan_tokens, self.chunk_size, self.base)
        seen_hashes: set[bytes] = set()
        results: List[BlendMatchResult] = []

        for rel_start, fingerprint in enumerate(rolling):
            candidate_ids = self._fingerprint_to_chunk_ids.get(fingerprint)
            if not candidate_ids:
                continue

            query_start = rel_start + scan_start
            query_chunk = tuple(
                int(token) for token in token_ids[query_start : query_start + self.chunk_size]
            )

            for compact_id in candidate_ids:
                storage_hash = self._chunk_storage_hashes[compact_id]
                stored_chunk = self._chunk_tokens[compact_id]
                if (
                    storage_hash is None
                    or stored_chunk is None
                    or storage_hash in seen_hashes
                    or stored_chunk != query_chunk
                ):
                    continue

                old_start = self._storage_hash_to_start.get(storage_hash)
                if old_start is None:
                    continue

                source_request_index = self._chunk_source_request_indexes[compact_id]
                first_seen_request_index = self._chunk_first_seen_request_indexes[compact_id]
                from_decode_cache = self._chunk_from_decode_cache[compact_id]
                if (
                    source_request_index is None
                    or first_seen_request_index is None
                    or from_decode_cache is None
                ):
                    continue

                results.append(
                    BlendMatchResult(
                        old_st=old_start,
                        old_ed=old_start + self.chunk_size,
                        cur_st=query_start,
                        cur_ed=query_start + self.chunk_size,
                        compact_id=compact_id,
                        fingerprint=fingerprint,
                        fingerprint_id=chunk_fingerprint_id(fingerprint),
                        storage_hash=storage_hash,
                        storage_hash_id=storage_hash_id(storage_hash),
                        source_request_index=source_request_index,
                        first_seen_request_index=first_seen_request_index,
                        from_decode_cache=from_decode_cache,
                    )
                )
                seen_hashes.add(storage_hash)
                break

        results.sort(key=lambda match: match.cur_st)
        return results

    def remove_chunks(self, storage_hashes: Iterable[bytes]) -> None:
        for storage_hash in storage_hashes:
            compact_id = self._storage_hash_to_compact_id.get(storage_hash)
            if compact_id is None:
                continue
            self._chunk_storage_hashes[compact_id] = None
            self._chunk_tokens[compact_id] = None
            self._chunk_fingerprints[compact_id] = None
            self._chunk_source_request_indexes[compact_id] = None
            self._chunk_first_seen_request_indexes[compact_id] = None
            self._chunk_from_decode_cache[compact_id] = None
            self._storage_hash_to_start.pop(storage_hash, None)
            self._storage_hash_to_compact_id.pop(storage_hash, None)
