# Non-Prefix KV Cache Reuse: Trace Collection and Analysis

A dataset and toolchain for measuring **non-prefix KV cache reusability** -- the kind of token reuse that standard prefix caching misses.

Standard prefix caching only reuses tokens that match from the start of a sequence. In multi-turn RAG, the same supporting documents often reappear across turns but at different positions -- retrieval order changes as the query evolves, new documents are added, and the conversation context shifts. These shared tokens are recomputed from scratch by prefix caching. CacheBlend-style systems can recover this waste by matching content-identical chunks regardless of position. This dataset quantifies how much waste there is to recover.

## Key Results

**110 sessions, 943 requests, 44.3M input tokens** across 4 domains from [MT-RAG benchmark](https://github.com/ibm/mt-rag-benchmark).

| Domain | Sessions | Requests | Prefix | Non-Prefix | New Compute |
|--------|:--------:|:--------:|-------:|-----------:|------------:|
| clapnq | 29 | 189 | 36.8% | **15.3%** | 47.9% |
| cloud | 26 | 251 | 56.2% | **17.1%** | 26.6% |
| govt | 28 | 271 | 54.1% | **15.1%** | 30.8% |
| fiqa | 27 | 232 | 80.8% | 3.5% | 15.7% |
| **Overall** | **110** | **943** | **54.9%** | **14.2%** | **31.0%** |

**14.2% of input tokens are reusable but invisible to prefix caching.** Recovering these requires content-based chunk matching (CacheBlend). See [`non_prefix_raw_traces/mtRag/README.md`](non_prefix_raw_traces/mtRag/README.md) for the full analysis, document-size threshold derivation, and workload comparison.

## Repository Structure

```
NonPrefix_LMCacheDataset/
├── non_prefix_raw_traces/             # Main dataset
│   └── mtRag/                         #   110 session JSONL files (4 domains) + README
├── raw_traces/                        # Earlier exploratory experiments
│   ├── ClaudeCode/                    #   Claude Code traces (compact + re-read)
│   ├── mtRag_on_OpenClaw/             #   Single-session MT-RAG through OpenClaw
│   ├── repo_on_OpenClaw/              #   Claude Code on LMCache repo via OpenClaw
│   ├── test_OpenClaw/                 #   OpenClaw behavior tests (resume, RAG, compaction)
│   └── README.md                      #   Summary of all experiments and findings
├── offline_analysis/                  # CacheBlend-style reuse analyzer
│   ├── analyze_trace.py               #   Core analyzer (prefix/non-prefix/new-compute)
│   ├── cacheblend_hashes.py           #   Two-hash engine (rolling prefix + content fingerprint)
│   ├── trace_viewer.html              #   Interactive HTML visualization template
│   └── README.md                      #   Usage and methodology
├── mtRag_traces_prompt_building/      # Prompt construction pipeline
│   ├── build_prefix_break_traces.py   #   Build session JSONL from MT-RAG corpus
│   ├── send_to_openclaw.py            #   Send sessions to OpenClaw with proxy management
│   ├── rebuild_capped.py              #   Rebuild with token cap enforcement
│   └── {clapnq,cloud,fiqa,govt}.jsonl #   Per-domain session prompts
└── proxy/                             # Recording proxies
    ├── OpenClaw_proxy.py              #   OpenClaw <-> vLLM recording proxy
    ├── anthropic_proxy.py             #   Claude Code <-> Anthropic API proxy
    └── openai_proxy.py                #   Generic OpenAI-format proxy
```

## How It Works

1. **Prompt construction** (`mtRag_traces_prompt_building/`): MT-RAG questions are grouped into multi-turn sessions. Each turn includes the question's supporting documents drawn in random order -- as they would be in a real retrieval system -- naturally producing non-prefix overlap between requests.

2. **Trace collection** (`proxy/`): Sessions are sent to OpenClaw (or vLLM directly) through a recording proxy that captures raw `/v1/chat/completions` traffic as JSONL with per-session trace files.

3. **Offline analysis** (`offline_analysis/`): CacheBlend-style chunk matching (tiktoken o200k_base, 256-token chunks) classifies each token as prefix-reusable, non-prefix-reusable, or new compute.

## Earlier Experiments

Before arriving at the multi-session MT-RAG approach, we tried several workloads. Most produced near-zero non-prefix reuse because they are inherently prefix-append. See [`raw_traces/README.md`](raw_traces/README.md) for details. Traces are in [`non_prefix_raw_traces/mtRag/`](non_prefix_raw_traces/mtRag/).

| Experiment | Sessions | Non-Prefix | Why low/high |
|------------|:--------:|:----------:|--------------|
| Claude Code with `/compact` | 1 | **10.5%** | Compaction rewrites history, shifting positions |
| Claude Code re-read after edit | 1 | 2.9% | Mostly prefix-append |
| MT-RAG single-session (4 domains) | 4 | 0.5-8.5% | History accumulates in order -- docs never move |
| Claude Code on LMCache repo via OpenClaw | 12 | 0.4% | Same prefix-append pattern |
| **Multi-session MT-RAG (this dataset)** | **110** | **14.2%** | **Shared docs at different positions across turns** |
