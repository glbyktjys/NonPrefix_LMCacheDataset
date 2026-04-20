# Raw Traces

Exploratory traces collected while investigating what kinds of workloads produce non-prefix KV cache reuse. These are the experiments that led to the final dataset in `shuffle_mtRag/`.

## Summary

| Experiment | Sessions | Requests | Non-Prefix Reuse | Notes |
|------------|:--------:|:--------:|:-----------------:|-------|
| `ClaudeCode/` (compact) | 1 | 30 | **10.5%** | `/compact` rewrites history, creating non-prefix overlap |
| `ClaudeCode/` (reRead) | 1 | 24 | 2.9% | Mostly prefix-append, fresh reads add small tail |
| `mtRag_on_OpenClaw/` (clapnq) | 1 | 214 | 0.5% | Single long session, prefix-dominated |
| `mtRag_on_OpenClaw/` (cloud) | 1 | 226 | 8.5% | Frequent compaction helped |
| `mtRag_on_OpenClaw/` (fiqa) | 1 | 186 | 0.8% | Small docs + single session |
| `mtRag_on_OpenClaw/` (govt) | 1 | 244 | 7.0% | Frequent compaction helped |
| `repo_on_OpenClaw/` | 12 | 1,299 | 0.4% | Claude Code on LMCache repo via OpenClaw |
| **`shuffle_mtRag/`** | **110** | **943** | **14.2%** | **Multi-session MT-RAG with randomized doc order** |

### What drives non-prefix reuse

Claude Code's `/compact` session (10.5%) and the multi-session MT-RAG traces (14.2%) both show meaningful non-prefix reuse -- but for different reasons:

- **Claude Code `/compact`**: compaction rewrites conversation history (summary replaces verbatim turns), so post-compaction requests share content with pre-compaction requests at different positions. The tool+system+conversation structure means tool definitions and system prompt remain stable while the history shifts.
- **Multi-session MT-RAG**: shared documents appear at different positions across turns because retrieval order changes naturally as the query evolves.

The experiments with near-zero non-prefix reuse (repo_on_OpenClaw at 0.4%, single-session mtRag at 0.5-0.8%) are all **prefix-append dominant** -- conversation history grows monotonically and shared content never moves.

---

## ClaudeCode/

Recorded Claude Code (Anthropic API) through a local logging proxy.

- `testing_compact_session_trace.jsonl` -- 30 reqs, 1.07M input tokens, **pfx=78.0%, np=10.5%**, new=11.5%. Six turns with 2 `/compact` triggers. Compaction rewrites the message array, breaking the prefix and creating non-prefix overlap.
- `testing_reRead_session_trace.jsonl` -- 24 reqs, 869K input tokens, pfx=87.8%, np=2.9%, new=9.4%. Four turns testing re-read after PR edit. Mostly prefix-append; fresh file reads add small non-prefix content.

## mtRag_on_OpenClaw/

All 777 MT-RAG questions run as 4 long single-session conversations through OpenClaw. 870 total backend requests.

| Domain | Reqs | Input Tokens | Prefix | Non-Prefix | New |
|--------|:----:|-------------:|-------:|-----------:|----:|
| clapnq | 214 | 18,901,360 | 99.0% | 0.5% | 0.5% |
| cloud | 226 | 19,453,412 | 83.0% | 8.5% | 8.5% |
| fiqa | 186 | 16,697,797 | 98.4% | 0.8% | 0.8% |
| govt | 244 | 24,003,323 | 85.9% | 7.0% | 7.0% |

Cloud and govt show higher non-prefix because they triggered more compaction episodes (14-16 episodes vs 1 for clapnq/fiqa), which breaks the prefix and leaves some verbatim content at shifted positions.

## repo_on_OpenClaw/

Claude Code working on the LMCache repository through OpenClaw, split into 12 topical sessions. 1,299 total requests, 71.8M input tokens. Overall: pfx=96.4%, **np=0.4%**. Same prefix-append pattern as direct Claude Code -- OpenClaw's conversation accumulation doesn't create position shifts.

## OpenClaw/

Manual OpenClaw conversations (TSLA 10-K, repeated-question tests, inline RAG experiments). Small-scale exploratory traces, not systematically analyzed.
