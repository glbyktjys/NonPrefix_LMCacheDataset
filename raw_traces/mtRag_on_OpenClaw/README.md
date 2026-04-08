# MT-RAG on OpenClaw

This folder contains raw traces and derived analysis for running the IBM MT-RAG benchmark through OpenClaw in long single-session mode.

## Goal

We want to stress OpenClaw's rolling context compaction and check whether it behaves like:

```text
turn 1: A B C D E F G
compact older prefix -> S1
next visible context: S1 + E F G

turn 2: S1 E F G H I
compact older prefix again -> S2
next visible context: S2 + G H I
```

In other words:
- old prefix is summarized
- direct prefix reuse breaks
- some recent old content is still preserved later in the prompt as non-prefix content

## Benchmark

MT-RAG is a multi-turn RAG benchmark with 4 domains:

- `clapnq`: 208 questions
- `fiqa`: 180 questions
- `govt`: 201 questions
- `cloud`: 188 questions

Total: `777` benchmark questions.

Each benchmark turn has:
- a question
- mapped retrieved corpus entries

For this experiment, the retrieved corpus content was inlined into the prompt, and each field was run as one persistent OpenClaw session to encourage context growth and compaction.

## What Was Run

- `clapnq` -> 1 OpenClaw session
- `fiqa` -> 1 OpenClaw session
- `govt` -> 1 OpenClaw session
- `cloud` -> 1 OpenClaw session

OpenClaw was configured to call a local proxy, which forwarded requests to the backend model server. We recorded the raw backend-facing `POST /v1/chat/completions` traffic.

## What Is In This Folder

- [traces](/Users/jiayuechen/Desktop/NonPrefix_LMCacheDataset/raw_traces/mtRag_on_OpenClaw/traces): raw JSONL traces
- [analysis](/Users/jiayuechen/Desktop/NonPrefix_LMCacheDataset/raw_traces/mtRag_on_OpenClaw/analysis): offline compaction summaries
- [visualizations](/Users/jiayuechen/Desktop/NonPrefix_LMCacheDataset/raw_traces/mtRag_on_OpenClaw/visualizations): HTML per-turn views
- [analyze_compaction.py](/Users/jiayuechen/Desktop/NonPrefix_LMCacheDataset/raw_traces/mtRag_on_OpenClaw/analyze_compaction.py): derives compaction episodes and summary-vs-tail structure
- [visualize_trace.py](/Users/jiayuechen/Desktop/NonPrefix_LMCacheDataset/raw_traces/mtRag_on_OpenClaw/visualize_trace.py): renders per-turn tables with prefix/non-prefix bars
- [split_trace_jsonl.py](/Users/jiayuechen/Desktop/NonPrefix_LMCacheDataset/raw_traces/mtRag_on_OpenClaw/split_trace_jsonl.py): splits large traces for easier manual inspection

## Key Results

Across the 4 fields:

- `870` total backend LLM requests in the raw traces
- `87` internal compaction requests
- `32` compaction episodes. A compaction episode is one logical compaction event, even if OpenClaw makes multiple internal compaction requests before the next visible turn. For example, the `clapnq` run has 1 episode made up of 6 internal compaction requests.
- `10 / 32` episodes matched the target pattern: prefix broken, recent old content still preserved as non-prefix


Per field:

| Field | Questions | Backend Requests | Internal Compaction Requests | Compaction Episodes | Episodes Matching `summary + recent tail` |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClapNQ | 208 | 214 | 6 | 1 | 1 |
| FiQA | 180 | 186 | 5 | 1 | 1 |
| Govt | 201 | 244 | 42 | 16 | 7 |
| Cloud | 188 | 226 | 34 | 14 | 1 |

## Takeaway

OpenClaw does compact in long MT-RAG sessions, but it does not always produce the same post-compaction shape.

- `clapnq` and `fiqa`: clear `summary + recent tail`
- `govt`: mixed behavior
- `cloud`: mostly more aggressive summary/reset behavior, with little retained recent tail
Our best explanation is that `cloud` often reaches compaction with a very large current context already in place. When compaction triggers, OpenClaw appears to compress more aggressively, so less of the recent pre-compaction context remains intact afterward.
    - In `cloud`, most compaction episodes collapse to a very small visible prompt after compaction.
    - The next normal prompt after a `cloud` compaction averages only about `3.7` messages, versus `20.8` messages before compaction.
    - This suggests that once the context becomes very large, the system keeps the summary but preserves less of the recent old tail conversation history.
    - By comparison, `clapnq` and `fiqa` each show one cleaner rewrite where a summary is inserted and a substantial recent tail is still preserved.
This is an inference from the trace structure, not a documented OpenClaw rule, but it matches what we observe in the raw prompts.

So the main conclusion is:

- compaction is real
- direct prefix reuse is often broken at the compaction boundary
- retained old content sometimes survives as non-prefix content, but not consistently across all domains

## Notes

- Token counts in the analysis and visualization are approximate. The raw traces do not include backend prompt usage, so token counts are estimated from normalized message text.

## View the Visualizer

Open the 4 session views directly:

```bash
open raw_traces/mtRag_on_OpenClaw/visualizations/openclaw_clapnq_session_trace.html
open raw_traces/mtRag_on_OpenClaw/visualizations/openclaw_fiqa_session_trace.html
open raw_traces/mtRag_on_OpenClaw/visualizations/openclaw_govt_session_trace.html
open raw_traces/mtRag_on_OpenClaw/visualizations/openclaw_cloud_session_trace.html
```
