"""
Build multi-turn RAG traces from mt-rag-benchmark's mtrag-human source,
shuffled per-turn to break prefix reuse while preserving document overlap.

One output JSONL file per domain; one line per conversation (a "session")
containing an ordered list of turns. Each turn's prompt carries the union of
gold documents seen so far in the conversation, permuted so the leading
document and individual positions differ from the previous turn.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text) // 4

# Leave room for system prompt (~6K) + compacted history (~2K) + output in 196K window
MAX_PROMPT_TOKENS = 180_000

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "mtrag-human" / "generation_tasks" / "reference.jsonl"
DOC_LVL_DIR = ROOT / "corpora" / "document_level" / "extracted"
OUT_DIR = ROOT.parent / "traces"

COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
}

RNG = random.Random(42)


def parse_doc_id(domain: str, passage_id: str) -> str:
    if domain == "clapnq":
        return passage_id.split("_", 1)[0]
    return passage_id.split("-", 1)[0]


def load_doc_texts(domain: str) -> dict:
    """Return {doc_id: full_text}. For clapnq, concatenate section chunks."""
    path = DOC_LVL_DIR / f"{domain}.jsonl"
    if domain == "clapnq":
        buckets = defaultdict(list)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                _id = d["_id"]
                doc_id, _, offsets = _id.partition("_")
                try:
                    start = int(offsets.split("-", 1)[0])
                except ValueError:
                    start = 0
                buckets[doc_id].append((start, d.get("title", ""), d.get("text", "")))
        out = {}
        for doc_id, chunks in buckets.items():
            chunks.sort()
            title = chunks[0][1]
            body = "\n".join(c[2] for c in chunks)
            out[doc_id] = (f"{title}\n{body}" if title else body).strip()
        return out

    out = {}
    key = "document_id" if domain in ("cloud", "govt") else "_id"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            title = d.get("title") or ""
            text = d.get("text") or ""
            out[d[key]] = (f"{title}\n{text}" if title else text).strip()
    return out


def load_conversations():
    """Group tasks by domain -> conversation_id -> sorted list of turn dicts."""
    by_domain = defaultdict(lambda: defaultdict(list))
    with open(REF) as f:
        for line in f:
            d = json.loads(line)
            domain = COLLECTION_TO_DOMAIN.get(d.get("Collection", ""))
            if not domain:
                continue
            question = next(
                (m["text"] for m in reversed(d["input"]) if m["speaker"] == "user"),
                "",
            )
            target = ""
            if d.get("targets"):
                target = d["targets"][0].get("text", "")
            by_domain[domain][d["conversation_id"]].append(
                {
                    "turn": int(d["turn"]),
                    "question": question,
                    "target": target,
                    "contexts": d.get("contexts", []),
                }
            )
    for dom in by_domain:
        for cid in by_domain[dom]:
            by_domain[dom][cid].sort(key=lambda t: t["turn"])
    return by_domain


def prefix_breaking_shuffle(docs: list, prev_order: list) -> list:
    """Random permutation where (when possible) the first doc and every carried
    doc's position differ from prev_order. Deterministic via seeded RNG."""
    if len(docs) <= 1:
        return list(docs)
    prev_index = {d: i for i, d in enumerate(prev_order)}
    best = None
    for _ in range(20):
        trial = docs[:]
        RNG.shuffle(trial)
        leading_ok = not prev_order or trial[0] != prev_order[0]
        positions_ok = all(
            i != prev_index.get(d, -1) for i, d in enumerate(trial)
        )
        if leading_ok and positions_ok:
            return trial
        if leading_ok and best is None:
            best = trial
    return best or trial


def render_prompt(doc_ids, passage_text_by_id, doc_map, domain, question,
                   max_tokens=MAX_PROMPT_TOKENS):
    """Render documents + question, dropping trailing docs if over token budget.

    Returns (prompt_text, kept_doc_ids) so the caller knows which docs survived.
    """
    if not doc_ids:
        return f"Question: {question}", []

    header = "Based on the following reference documents, answer the question at the end.\n\n"
    q_line = f"Question: {question}"
    budget = max_tokens - _count_tokens(header) - _count_tokens(q_line)

    kept_ids = []
    doc_blocks = []
    running = 0
    for pid in doc_ids:
        doc_id = parse_doc_id(domain, pid)
        text = doc_map.get(doc_id) or passage_text_by_id.get(pid, "")
        block = f"[Document {len(doc_blocks) + 1}: {text}]\n"
        block_tokens = _count_tokens(block)
        if running + block_tokens > budget:
            break
        running += block_tokens
        doc_blocks.append(block)
        kept_ids.append(pid)

    prompt = header + "\n".join(doc_blocks) + "\n" + q_line
    return prompt, kept_ids


def shuffle_metrics(sessions):
    leading_diff = 0
    leading_total = 0
    pos_changes = 0
    pos_total = 0
    for s in sessions:
        turns = s["turns"]
        for i in range(1, len(turns)):
            prev = turns[i - 1]["document_ids"]
            cur = turns[i]["document_ids"]
            if prev and cur:
                leading_total += 1
                if prev[0] != cur[0]:
                    leading_diff += 1
            prev_idx = {d: j for j, d in enumerate(prev)}
            for j, d in enumerate(cur):
                if d in prev_idx:
                    pos_total += 1
                    if prev_idx[d] != j:
                        pos_changes += 1
    return {
        "leading_diff_rate": leading_diff / leading_total if leading_total else 0,
        "position_change_rate": pos_changes / pos_total if pos_total else 0,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    convs_by_domain = load_conversations()
    for domain, convs in convs_by_domain.items():
        print(f"\n=== {domain}: {len(convs)} conversations ===")
        doc_map = load_doc_texts(domain)
        print(f"  loaded {len(doc_map)} document_level docs")
        out_path = OUT_DIR / f"{domain}.jsonl"
        sessions = []
        total_turns = 0
        fallbacks = 0
        total_slots = 0
        with open(out_path, "w") as out_f:
            for conv_id, turns in convs.items():
                accumulated = []
                accumulated_set = set()
                prev_order = []
                session_turns = []
                passage_text_by_id = {}
                for t in turns:
                    for c in t["contexts"]:
                        passage_text_by_id[c["document_id"]] = c.get("text", "")
                        if c["document_id"] not in accumulated_set:
                            accumulated_set.add(c["document_id"])
                            accumulated.append(c["document_id"])
                    shuffled = prefix_breaking_shuffle(accumulated, prev_order)
                    for pid in shuffled:
                        total_slots += 1
                        if parse_doc_id(domain, pid) not in doc_map:
                            fallbacks += 1
                    prompt, kept_ids = render_prompt(
                        shuffled, passage_text_by_id, doc_map, domain, t["question"]
                    )
                    if len(kept_ids) < len(shuffled):
                        dropped = len(shuffled) - len(kept_ids)
                        print(f"    {domain}_{conv_id} turn {t['turn']}: "
                              f"capped {len(shuffled)} → {len(kept_ids)} docs "
                              f"(-{dropped})")
                    session_turns.append({
                        "metadata": f"{domain}_{conv_id}_turn{t['turn']}",
                        "turn": t["turn"],
                        "document_ids": kept_ids,
                        "prompt": prompt,
                    })
                    prev_order = shuffled
                total_turns += len(session_turns)
                session = {
                    "session_id": f"{domain}_{conv_id}",
                    "domain": domain,
                    "turns": session_turns,
                }
                out_f.write(json.dumps(session) + "\n")
                sessions.append(session)
        m = shuffle_metrics(sessions)
        print(f"  wrote {out_path.name}: {len(sessions)} sessions, {total_turns} turns")
        print(
            f"  leading_diff_rate={m['leading_diff_rate']:.3f} "
            f"position_change_rate={m['position_change_rate']:.3f} "
            f"doc_fallbacks={fallbacks}/{total_slots}"
        )


if __name__ == "__main__":
    main()
