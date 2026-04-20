# LMCache Non-Prefix Traces Collection

We explored prefix-breaking behavior (non-prefix reuse) in agent settings, from Claude Code to OpenClaw, to collect realistic traces that motivate CacheBlend-style content-based KV cache reuse.

## Background

LMCache's prefix caching works well when each request is a strict extension of the previous one. But in multi-turn agent workflows, context can be rewritten, compacted, or reconstructed — breaking the prefix and leaving reusable content at new positions. CacheBlend's two-hash design (position-dependent storage hash + position-independent content fingerprint) can recover this "non-prefix reusable" content. We need traces that demonstrate where this matters.

## Core Insight

The reason non-prefix reuse is low in most agent traces isn't that there's no content overlap — there IS. It's that **prefix-append conversation history** makes the overlapping content redundant. The original content is already cached in the prefix. Any re-inserted copy at a new position is technically non-prefix reusable, but it's a small fraction of the growing total.

To get high non-prefix reuse, you need either:
1. **Context compaction** — old prefix is summarized, breaking prefix continuity, but recent content survives at shifted positions
2. **Stateless retrieval** — each request is reconstructed from scratch with retrieved documents (not how OpenClaw/Claude Code work natively)

---

## Questions Investigated and Results

### Q1: Does OpenClaw re-read files when asked the same question twice?

**Action:** Sent 3 questions to OpenClaw — Q1 (original), Q2 (identical to Q1), Q3 (slight rewording). Collected trace: `raw_traces/OpenClaw/same_question_test_trace.jsonl`.

**Result:** OpenClaw answers identical questions from conversation history without re-reading. Q2 issued zero tool calls because Q1's tool results were already in the prefix. Q3 (slight rewording) triggered 10 new tool calls. This confirms the prefix-append model: there is no mechanism for the same content to appear at a different position unless compaction occurs.

**Trace produced:** `same_question_test_trace.jsonl` (30 records, 3 questions)

---

### Q2: Does mem0 RAG retrieval in OpenClaw produce non-prefix reuse?

**Hypothesis:** If mem0 retrieves the same documents for different questions, the shared retrieved content would appear at different positions across turns, creating non-prefix reuse.

**Action:**
1. Added 3 Cotton gin-related documents to mem0 via `mem0/addMemory.py` (608 tokens total across 3 docs)
2. Enabled the `openclaw-mem0` plugin with `apiKey` and `userId` config
3. Asked 2 questions that should retrieve overlapping documents
4. Collected trace: `raw_traces/OpenClaw/rag_test_trace.jsonl`

**Result:** 0% non-prefix reuse. Two problems:
- **mem0 infers/summarizes content** even with `infer=False` (the platform API ignores this flag). The 3 original documents (608 tokens) were compressed into 8 bullet-point summaries (251 tokens total, 41.3% of original). The longest bullet was only 48 tokens.
- **CacheBlend requires 256 contiguous matching tokens** to form one chunk. All mem0 summaries are far below this threshold, making non-prefix matching mathematically impossible.

**Trace produced:** `rag_test_trace.jsonl` (6 records, 2 questions), `no_infer_rag_test_trace.jsonl`

---

### Q3: Does inline RAG (documents embedded directly in prompts) produce non-prefix reuse?

**Hypothesis:** If we bypass mem0 and inline full documents directly into user messages, shared documents across turns should produce non-prefix reuse.

**Action:** Manually constructed prompts with overlapping Cotton gin documents: Q1 = [doc1] + [doc2] + question, Q2 = [doc2] + [doc3] + question. Collected trace: `raw_traces/OpenClaw/in_line_RAG_test_trace.jsonl`.

**Result:** 0% non-prefix reuse. The shared document (Cotton gin, 227 tokens) is below CacheBlend's 256-token chunk size, so no complete chunk can match. Additionally, the prefix-append model means the original doc from Q1 is already in the prefix of Q2's request — the re-inserted copy is redundant.

**Trace produced:** `in_line_RAG_test_trace.jsonl` (6 records, 2 questions)

---

### Q4: Does MT-RAG benchmark with many overlapping documents produce meaningful non-prefix reuse?

**Hypothesis:** A large-scale benchmark (777 questions across 4 domains) with 335/498 shared corpus IDs across turns should produce detectable non-prefix reuse, especially with longer documents.

**Action:**
1. Used IBM MT-RAG benchmark: 4 domains (clapnq 208q, fiqa 180q, govt 201q, cloud 188q)
2. Inlined retrieved corpus content into each prompt
3. Ran each domain as one long persistent OpenClaw session to encourage compaction
4. Collected traces via recording proxy

**Result:** Non-prefix reuse stayed at 0.1–1.2% across all domains. The prefix-append model dominates — shared documents are already in the prefix from earlier turns. Non-prefix reuse only appears at compaction boundaries.

| Session | Questions | Prefix Ratio | Non-Prefix Ratio | New Compute |
|---------|----------:|-------------:|-----------------:|------------:|
| clapnq  | 208       | 98.37%       | 0.11%            | 1.52%       |
| fiqa    | 180       | 97.84%       | 0.12%            | 2.04%       |
| govt    | 201       | 84.95%       | 1.16%            | 13.89%      |
| cloud   | 188       | 79.05%       | 0.79%            | 20.16%      |
| **Combined** | **777** | **89.43%** | **0.60%**      | **9.97%**   |

**Compaction analysis:** 870 total backend requests, 32 compaction episodes detected, but only 10/32 matched the target pattern (prefix broken + recent tail preserved as non-prefix content).

**Traces produced:** `raw_traces/mtRag_on_OpenClaw/traces/openclaw_{clapnq,fiqa,govt,cloud}_session_trace.jsonl`

---

### Q5: Does OpenClaw's compaction actually produce non-prefix reuse like Claude Code's /compact?

**Action:** Analyzed OpenClaw compaction behavior across both MT-RAG traces (777 questions) and LMCache PR traces (100 questions in single session + 11 grouped sessions).

**Result:** OpenClaw's compaction silently fails in safeguard mode (the default) — there are multiple open issues about this (#7477, #3436, #15669). Instead of generating an LLM summary like Claude Code does, it just truncates and retries after context overflow. The `[Chat messages since your last reply...]` we saw in traces isn't real compaction — it's OpenClaw's fallback behavior. This is why we see ~0% non-prefix reuse in OpenClaw.

**Claude Code's compaction is structurally different:** it's a visible API call — a compact agent with different tools (0–2 vs 30+) and different system prompt. This creates shifted blocks that CacheBlend can actually reuse.

| System | Single Session | Prefix | Non-Prefix | New Compute | Compaction Events |
|--------|---------------|--------|------------|-------------|-------------------|
| OpenClaw (100 LMCache PRs) | 496 requests | 98.2% | 0.42% | 1.4% | 49 (truncation-based) |
| OpenClaw (11 grouped) | varies | 95–99% | <0.5% | 1–5% | minimal |
| Claude Code | 53 requests | high within segments | ~10% at /compact | varies | real LLM summary |

**Traces produced:** `raw_traces/repo_on_OpenClaw/LMCache_*.jsonl` (13 session traces), `raw_traces/ClaudeCode/testing_compact_session_trace.jsonl`

---

### Q6: Can we build larger inline RAG prompts and send them through OpenClaw to collect non-prefix traces at scale?

**Action:**
1. Built 4 JSONL files from MT-RAG benchmark with inline documents embedded in prompts:
   - `traces_prompt_building/clapnq.jsonl` (29 sessions, 224 turns)
   - `traces_prompt_building/cloud.jsonl` (26 sessions, 205 turns)
   - `traces_prompt_building/fiqa.jsonl` (27 sessions, 199 turns)
   - `traces_prompt_building/govt.jsonl` (28 sessions, 214 turns)
2. Built `traces_prompt_building/send_to_openclaw.py` to send these as multi-turn conversations via `/v1/chat/completions`, with automatic proxy session start/end per session and Bearer token auth

**Status:** Script ready, pending execution on remote server.

**Key design:** The `/v1/chat/completions` API is stateless — conversation context is entirely in the `messages` array. The script resets messages for each new session_id (= new conversation). The proxy's `/session/start` and `/session/end` are called automatically between sessions for separate trace files.

---

## Two Practical Paths for Generating Non-Prefix Traces

### Path 1: RAG-style benchmarks (inline documents)
The key pattern is that the same or overlapping retrieved chunks reappear across turns, but at different positions in the prompt. MT-RAG is a strong fit — later turns can reuse earlier context chunks without common prefixes. However, the prefix-append model limits effectiveness unless sessions are kept short or documents are very long (500+ tokens each).

### Path 2: Agent context compaction
Find agents where compaction can be triggered (e.g., Claude Code's `/compact`). Compaction creates append-with-reset behavior: the old prefix is summarized, direct prefix breaks, but recent content survives at shifted positions as non-prefix reusable content.

- **Claude Code /compact** is the most reliable trigger — produces ~10% non-prefix reuse at compaction boundaries
- **OpenClaw compaction** does not work as expected (truncation-based, not summary-based)

---

## Repository Structure

```
NonPrefix_LMCacheDataset/
├── raw_traces/
│   ├── OpenClaw/              # 9 trace files (same-question, RAG, inline-RAG, resume, compaction tests)
│   ├── mtRag_on_OpenClaw/     # 4 full MT-RAG traces + 12 splits (777 questions, 4 domains)
│   ├── repo_on_OpenClaw/      # 13 LMCache PR traces (100 questions, single + 11 grouped sessions)
│   └── ClaudeCode/            # 2 Claude Code traces (compact + re-read tests)
├── offline_analysis/
│   ├── analyze_trace.py       # Core analyzer (prefix/non-prefix/new-compute computation)
│   ├── cacheblend_hashes.py   # CacheBlend two-hash implementation (rolling prefix + polynomial fingerprint)
│   └── trace_viewer.html      # Interactive trace visualization template
├── cache_simulator/
│   ├── simulator.py           # LRU cache hit-rate simulator
│   ├── lru_cache.py           # LRU cache implementation
│   └── export_lookup_hashes.py
├── traces_prompt_building/
│   ├── send_to_openclaw.py    # Send inline-RAG sessions to OpenClaw with proxy session management
│   ├── build_prefix_break_traces.py
│   └── {clapnq,cloud,fiqa,govt}.jsonl  # MT-RAG inline prompts (110 sessions, 842 turns)
├── OpenClaw_trace_collection/
│   ├── collect_single_session.py
│   ├── collect_grouped_traces.py
│   └── questions/             # 10 question sets (LMCache, DeepSpeed, vLLM, etc.)
├── OpenClaw_trace_analysis/   # Analysis JSONs + HTML visualizations for all LMCache traces
├── proxy/                     # Recording proxies (OpenClaw, Anthropic, OpenAI)
└── mem0/                      # mem0 document ingestion scripts
```

## Tools Built

| Tool | Purpose |
|------|---------|
| `offline_analysis/analyze_trace.py` | Offline trace analyzer supporting both Anthropic and OpenAI formats. Computes prefix/non-prefix/new-compute using CacheBlend-style rolling hash fingerprinting with 256-token chunks. |
| `offline_analysis/cacheblend_hashes.py` | CacheBlend two-hash implementation: position-dependent rolling prefix hash (storage key) + position-independent polynomial fingerprint (content matching). |
| `offline_analysis/trace_viewer.html` | Interactive HTML dashboard for visualizing per-request chunk breakdown (prefix/non-prefix/new). Supports chunk-level selection. |
| `proxy/OpenClaw_proxy.py` | Recording proxy between OpenClaw and vLLM backend. Captures raw `/v1/chat/completions` traffic as JSONL with session management. |
| `proxy/anthropic_proxy.py` | Recording proxy for Claude Code to Anthropic API traffic. |
| `cache_simulator/simulator.py` | LRU cache hit-rate simulator. Replays lookup-hash events and computes token-level prefix hit rates. |
| `traces_prompt_building/send_to_openclaw.py` | Sends inline-RAG multi-turn sessions to OpenClaw with automatic proxy session start/end and Bearer token auth. |
