"""
Rebuild clapnq.jsonl from clapnq_original.jsonl with:
  1. Per-turn token cap (180K) to fit in 196K context window
  2. When capping, prioritize keeping shared documents (seen in prior turns)
     over new-to-this-turn documents, since shared docs create non-prefix reuse.
"""

import json
import re
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4

MAX_PROMPT_TOKENS = 140_000  # leave 56K for system prompt + compacted history + OpenClaw overhead

BASE = Path(__file__).resolve().parent
DOMAINS = ["clapnq", "cloud", "govt"]  # fiqa excluded: docs too small, already fits

# Pattern to extract individual document blocks from prompt text
DOC_PATTERN = re.compile(
    r'\[Document \d+:\s*(.*?)\](?=\s*\[Document \d+:|\s*Question:)',
    re.DOTALL,
)


def rebuild_session(session: dict, prev_session_doc_ids: set | None = None) -> dict:
    turns = sorted(session["turns"], key=lambda t: t["turn"])
    new_turns = []
    seen_doc_ids: set[str] = set()  # all doc IDs seen in prior turns of this session

    for turn in turns:
        doc_ids = turn["document_ids"]
        prompt = turn["prompt"]

        if not doc_ids:
            # No documents (e.g. bare question), keep as-is
            new_turns.append(turn)
            continue

        # Extract each doc's text from the prompt
        doc_texts = DOC_PATTERN.findall(prompt)

        # Match doc_ids to doc_texts (they're in the same order)
        if len(doc_texts) != len(doc_ids):
            # Fallback: can't reliably match, keep as-is
            seen_doc_ids.update(doc_ids)
            new_turns.append(turn)
            continue

        # Build list of (doc_id, text, token_count, is_shared)
        docs = []
        for pid, text in zip(doc_ids, doc_texts):
            tok = count_tokens(f"[Document X: {text}]\n")
            is_shared = pid in seen_doc_ids
            docs.append((pid, text.strip(), tok, is_shared))

        # Compute budget
        q_match = re.search(r'Question: .+$', prompt, re.DOTALL)
        question_text = q_match.group(0) if q_match else ""
        header = "Based on the following reference documents, answer the question at the end.\n\n"
        budget = MAX_PROMPT_TOKENS - count_tokens(header) - count_tokens(question_text)

        total_tokens = sum(d[2] for d in docs)

        if total_tokens <= budget:
            # Fits as-is, no capping needed
            seen_doc_ids.update(doc_ids)
            new_turns.append(turn)
            continue

        # Need to cap — prioritize shared docs
        # Split into shared (reuse value) and new (no reuse value)
        shared_docs = [(i, d) for i, d in enumerate(docs) if d[3]]
        new_docs = [(i, d) for i, d in enumerate(docs) if not d[3]]

        # Greedily add: all shared docs first, then new docs
        kept_indices = set()
        running = 0

        # Add shared docs (these create non-prefix reuse)
        for i, d in shared_docs:
            if running + d[2] > budget:
                break
            running += d[2]
            kept_indices.add(i)

        # Fill remaining budget with new docs
        for i, d in new_docs:
            if running + d[2] > budget:
                break
            running += d[2]
            kept_indices.add(i)

        # Rebuild prompt preserving original order
        kept_doc_ids = []
        doc_blocks = []
        doc_num = 1
        for i, (pid, text, tok, is_shared) in enumerate(docs):
            if i in kept_indices:
                doc_blocks.append(f"[Document {doc_num}: {text}]")
                kept_doc_ids.append(pid)
                doc_num += 1

        new_prompt = header + "\n".join(doc_blocks) + "\n\n" + question_text

        dropped_shared = sum(1 for i, d in shared_docs if i not in kept_indices)
        dropped_new = sum(1 for i, d in new_docs if i not in kept_indices)
        total_dropped = len(docs) - len(kept_indices)
        print(f"    turn {turn['turn']}: {len(docs)} → {len(kept_indices)} docs "
              f"(-{total_dropped}: {dropped_new} new, {dropped_shared} shared) "
              f"{count_tokens(new_prompt):,} tokens")

        seen_doc_ids.update(doc_ids)
        new_turns.append({
            "metadata": turn["metadata"],
            "turn": turn["turn"],
            "document_ids": kept_doc_ids,
            "prompt": new_prompt,
        })

    return {
        "session_id": session["session_id"],
        "domain": session.get("domain", "clapnq"),
        "turns": new_turns,
    }


def process_domain(domain: str):
    input_path = BASE / f"{domain}_original.jsonl"
    output_path = BASE / f"{domain}.jsonl"

    if not input_path.exists():
        print(f"Skipping {domain}: {input_path.name} not found")
        return

    sessions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"{domain}: {len(sessions)} sessions from {input_path.name}")
    print(f"Token cap: {MAX_PROMPT_TOKENS:,} per turn")
    print(f"{'='*60}")

    total_turns = 0
    capped_turns = 0
    total_docs_original = 0
    total_docs_kept = 0
    total_shared_original = 0
    total_shared_kept = 0

    rebuilt = []
    for session in sessions:
        sid = session["session_id"]
        turns_orig = sorted(session["turns"], key=lambda t: t["turn"])

        seen = set()
        for t in turns_orig:
            dids = t["document_ids"]
            for d in dids:
                if d in seen:
                    total_shared_original += 1
                total_docs_original += 1
            seen.update(dids)

        print(f"Session: {sid} ({len(turns_orig)} turns)")
        new_session = rebuild_session(session)
        rebuilt.append(new_session)

        new_turns = sorted(new_session["turns"], key=lambda t: t["turn"])
        seen = set()
        for t in new_turns:
            total_turns += 1
            dids = t["document_ids"]
            for d in dids:
                if d in seen:
                    total_shared_kept += 1
                total_docs_kept += 1
            seen.update(dids)

            orig_turn = next(ot for ot in turns_orig if ot["turn"] == t["turn"])
            if len(t["document_ids"]) < len(orig_turn["document_ids"]):
                capped_turns += 1

    with open(output_path, "w") as f:
        for s in rebuilt:
            f.write(json.dumps(s) + "\n")

    print(f"\n  Written to: {output_path.name}")
    print(f"  Sessions: {len(rebuilt)}")
    print(f"  Turns capped: {capped_turns}/{total_turns}")
    if total_docs_original > 0:
        print(f"  Documents: {total_docs_kept}/{total_docs_original} kept "
              f"({100*total_docs_kept/total_docs_original:.1f}%)")
    if total_shared_original > 0:
        print(f"  Shared docs: {total_shared_kept}/{total_shared_original} kept "
              f"({100*total_shared_kept/total_shared_original:.1f}%)")
        print(f"  → Non-prefix reuse potential preserved: "
              f"{100*total_shared_kept/total_shared_original:.0f}%")


def main():
    for domain in DOMAINS:
        process_domain(domain)


if __name__ == "__main__":
    main()
