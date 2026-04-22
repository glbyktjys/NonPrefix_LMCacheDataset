# Non-Prefix KV Cache Reuse: Trace Collection and Analysis

## What This Is

A dataset of multi-turn RAG traces designed to measure **non-prefix KV cache reusability** -- the kind of token reuse that standard prefix caching misses.

Standard prefix caching only reuses tokens that match from the start of a sequence. In multi-turn RAG, the same supporting documents often reappear across turns but at different positions -- retrieval order changes as the query evolves, new documents are added, and the conversation context shifts. These shared tokens are recomputed from scratch by prefix caching. CacheBlend-style systems can recover this waste by matching content-identical chunks regardless of position. This dataset quantifies how much waste there is to recover.

Source: [MT-RAG benchmark](https://github.com/ibm/mt-rag-benchmark) (mtrag-human), document-level corpus. Each session is a multi-turn RAG conversation where supporting documents accumulate across turns. Documents are drawn in random order per turn -- as they would be in a real retrieval system where ranking changes as the query evolves -- naturally producing non-prefix overlap between requests.

---

## 1. Results

**110 sessions, 943 requests, 44.3M input tokens** across 4 domains.

| Domain | Sessions | Requests | Input Tokens | Avg Reqs/Session |
|--------|:--------:|:--------:|-------------:|-----------------:|
| clapnq | 29 | 189 | 9,476,500 | 6.5 |
| cloud | 26 | 251 | 14,568,971 | 9.7 |
| govt | 28 | 271 | 14,049,565 | 9.7 |
| fiqa | 27 | 232 | 6,246,143 | 8.6 |
| **Total** | **110** | **943** | **44,341,179** | **8.6** |

### Reuse Breakdown (CacheBlend chunk size = 256 tokens)

| Domain | Prefix | Non-Prefix | New Compute | NP Median | NP Max |
|--------|-------:|-----------:|------------:|----------:|-------:|
| clapnq | 36.8% | **15.3%** | 47.9% | 12.6% | 31.8% |
| cloud | 56.2% | **17.1%** | 26.6% | 19.2% | 36.9% |
| govt | 54.1% | **15.1%** | 30.8% | 14.5% | 27.7% |
| fiqa | 80.8% | 3.5% | 15.7% | 2.4% | 6.7% |
| **Overall** | **54.9%** | **14.2%** | **31.0%** | 11.5% | 36.9% |

- **Prefix** = tokens reusable by standard prefix caching (system prompt, workspace context prepended identically each turn)
- **Non-prefix** = tokens reusable only by content-matching systems like CacheBlend (shared documents at different positions across turns)
- **New compute** = tokens that must be computed from scratch

### Why fiqa Is Low

The non-prefix gap between fiqa (3.5%) and the other three domains (15-17%) is explained entirely by document size. CacheBlend matches in 256-token chunks; documents smaller than 2 x chunk_size produce zero matchable chunks.

| Domain | Median Doc Size | Docs < 512 tok | Docs >= 2048 tok | Observed NP Reuse |
|--------|----------------:|---------------:|-----------------:|------------------:|
| clapnq | 7,417 | 0% | 97% | **15.3%** |
| cloud | 2,614 | 6% | 61% | **17.1%** |
| govt | 2,622 | 1% | 62% | **15.1%** |
| fiqa | 174 | 87% | 1% | 3.5% |

The ~15% average is not a ceiling on the method -- it reflects the document size distribution in this benchmark. Datasets with uniformly large documents would show higher non-prefix reuse. See [Section 3](#3-document-size-threshold) for the full threshold analysis.

---

## 2. What Workloads Favor Non-Prefix Reuse

Not all workloads benefit. Non-prefix reuse requires three conditions:

1. **Shared content across turns** -- the same documents, code blocks, or context appearing in multiple requests
2. **Position changes** -- the shared content appears at different offsets (different retrieval order, new documents inserted, reordered context)
3. **Documents large enough to match** -- each document must be significantly larger than the chunk size (>= 8C recommended; see [Section 3](#3-document-size-threshold))

Workloads that naturally exhibit this:
- Multi-turn RAG (documents accumulate and reorder across turns)
- Long coding sessions with file context (same files referenced at different conversation depths)

Workloads that do NOT benefit:
- Single-turn inference (no history to reuse)
- Pure prefix-append conversations (standard prefix caching already captures everything)
- Small-context chat (not enough tokens to form matchable chunks)

### Comparison with SemiAnalysis InferenceX Benchmark

The [SemiAnalysis InferenceX agentic benchmark](https://github.com/SemiAnalysisAI/InferenceX) (739 anonymized Claude Code traces on AMD MI300X) demonstrated that LMCache delivers **3x lower TTFT** and **2.3x more requests** under stress. That benchmark is an excellent demonstration of cache **capacity** -- LMCache's DRAM tier holds 36% more working set than HBM-only. But it measures a fundamentally different reuse pattern:

| | InferenceX Agentic Benchmark | This Dataset |
|---|---|---|
| **Workload** | Claude Code coding sessions | Multi-turn RAG (MT-RAG) |
| **Reuse pattern** | Prefix-append (each turn extends the previous) | Non-prefix (shared docs at different positions) |
| **Prefix overlap** | 93-97% per turn | 36-81% (system prompt + workspace only) |
| **Non-prefix reuse** | ~0% (all reuse is prefix) | **14.2% average** |
| **What captures it** | Standard prefix caching (HBM or DRAM) | Requires CacheBlend-style content matching |

Their traces are pure prefix-append -- every turn builds on the previous, so standard prefix caching captures nearly all reusable tokens. Our traces expose the complementary gap: **14.2% of input tokens are reusable but invisible to prefix caching**. Recovering these requires content-based chunk matching, which is what CacheBlend adds on top of the existing prefix + capacity stack.

---

## 3. Document Size Threshold

CacheBlend matches content in fixed-size chunks (C = 256 tokens). When a shared document appears at a different position across turns, only the **interior** tokens produce matchable chunks. The first and last ~C tokens are "contaminated" by surrounding context (document numbering, adjacent content) that differs across turns.

```
matchable(S) = max(0, S - 2C)
efficiency(S) = max(0, 1 - 2C / S)
```

| Tier | Document Size | Matchable Chunks | Efficiency | Verdict |
|------|:-------------|:----------------|:-----------|:--------|
| Zero | < 512 (2C) | 0 | 0% | Not viable |
| Marginal | 512 - 1024 (2-4C) | 1-2 | < 50% | Barely detectable |
| Viable | 1024 - 2048 (4-8C) | 2-6 | 50-75% | Practical minimum |
| **Good** | **>= 2048 (8C)** | **6+** | **> 75%** | **Recommended** |

**Rule of thumb: document size >= 8 x chunk_size for meaningful non-prefix reuse.**

---

## 4. Trace Format and Pipeline

### Trace Structure

Each trace file is a JSONL recording of HTTP requests from OpenClaw to the vLLM backend hosting MiniMax-M2.5, one file per session:

```
non_prefix_raw_traces/mtRag/
  clapnq_clapnq_{session_hash}_trace.jsonl   (29 files)
  cloud_cloud_{session_hash}_trace.jsonl      (26 files)
  govt_govt_{session_hash}_trace.jsonl        (28 files)
  fiqa_fiqa_{session_hash}_trace.jsonl        (27 files)
```

Each JSONL entry is a `request` or `response` record:

```json
{"type": "request", "request_id": "...", "timestamp_rel_s": 5.3,
 "method": "POST", "path": "/v1/chat/completions",
 "headers": {...}, "body": {"model": "MiniMaxAI/MiniMax-M2.5", "messages": [...]}}

{"type": "response", "request_id": "...", "timestamp_rel_s": 8.3,
 "status_code": 200, "body": {"choices": [...], "usage": {...}}}
```

Within each session, the request count is higher than the number of user turns because OpenClaw generates additional requests for tool calls and compaction summaries. A typical 8-turn session produces 10-20 backend requests:
- 1 `/new` reset
- N turn requests (messages accumulate: 2 -> 4 -> 6 -> ... -> 2N)
- 2 compaction summarization calls (when history exceeds context budget)
- 0-4 tool-call rounds (OpenClaw attempts web search, file reads, etc.)

### Collection Pipeline

```
MT-RAG JSONL (shuffled docs per turn)
    --> send_to_openclaw.py --> OpenClaw (port 18789)
        --> Recording Proxy (port 18790) --> vLLM + MiniMax M2.5 (port 8200)
                                                |
                                           trace JSONL files
```

- **Model**: MiniMax M2.5 (196K context, FP8 MoE, 4x H100 TP=4)
- **Middleware**: OpenClaw 2026.4.15 with safeguard compaction (reserveTokensFloor=80K)
- **Token cap**: 140K tokens per turn prompt, shared-document priority when capping
- **Analysis**: offline CacheBlend-style chunk matching (tiktoken o200k_base, chunk size 256)

---

## 5. Next Steps

- **Scale to 500-1,000 sessions.** Current collection covers 110 sessions. Targeting 500+ for statistical robustness, with additional domains and longer documents.
- **Convert to LMCache trace format.** Adapt traces for direct integration with LMCache benchmarking infrastructure.
- **Benchmark CacheBlend end-to-end.** Run traces through LMCache with CacheBlend enabled to measure actual TTFT improvement (not just theoretical token counts).
- **Explore smaller chunk sizes.** Reducing chunk size from 256 to 128 or 64 would lower the document-size threshold, potentially unlocking non-prefix reuse for fiqa-style corpora at the cost of higher hash overhead.
- **Wrap-up blog post.** Synthesize findings into a public write-up.

## 6. Earlier Experiments (`raw_traces/`)

Before arriving at the multi-session MT-RAG approach, we tried several other workloads. Most produced near-zero non-prefix reuse because they are inherently prefix-append -- each turn extends the previous message array without repositioning shared content.

| Experiment | Sessions | Requests | Non-Prefix | Why |
|------------|:--------:|:--------:|:----------:|-----|
| Claude Code with `/compact` | 1 | 30 | 10.5% | Compaction rewrites history, shifting content positions |
| Claude Code re-read after edit | 1 | 24 | 2.9% | Mostly prefix-append; fresh reads add small tail |
| MT-RAG single-session (4 domains) | 4 | 870 | 0.5-8.5% | History accumulates in order -- shared docs never move |
| Claude Code on LMCache repo via OpenClaw | 12 | 1,299 | 0.4% | Same prefix-append pattern through OpenClaw |

The single-session MT-RAG runs are the clearest contrast: the same documents and questions produce 0.5% non-prefix reuse when sent as one long session (prefix-append) vs **14.2%** when sent as separate sessions with randomized document order. The content is identical -- only the position changes.