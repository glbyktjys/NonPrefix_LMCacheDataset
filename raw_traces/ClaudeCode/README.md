# Claude Code Logging Proxy Plan

## Goal

Route local Claude Code through a custom proxy, log Anthropic API traffic, and convert the raw traces into a replayable dataset for studying context reuse patterns.

The two working goals are:

1. understand when Claude Code follows prefix-append behavior versus when it rewrites context
2. collect realistic traces that are useful for studying non-prefix reuse and motivating CacheBlend-style evaluation

This matters for prefix reuse:
- append-like behavior is more prefix-cache-friendly
- middle-of-prompt mutation reduces reusable prefix and motivates approaches like CacheBlend

### Early Finding

The earliest observation from a simple pre-`/compact` session was that Claude Code behaves like pure prefix append with a sliding prompt-cache marker.

Concretely, for each new turn N, the messages array sent to the API is exactly:

```
[msg_0, msg_1, ..., msg_{N-2}, msg_{N-1}(new)]
```

where `msg_0 … msg_{N-2}` are bit-for-bit identical to the previous request, except for one mutation: the `cache_control: {type: "ephemeral"}` marker is **removed** from the previous last message and **placed on the new last message**. This marker always sits at the tail of the array, acting as a sliding "cache up to here" pointer.

Key observations from `testing_session_trace.jsonl` (76 raw entries, 25 sonnet iterations):

| Observation | Detail |
|---|---|
| History strategy | Pure append — every earlier message is preserved verbatim |
| Only mutation | `cache_control` marker slides from msg[N-2] → msg[N-1] each turn |
| Message growth | Exactly +2 per user turn (user message + assistant response) |
| Sub-agent calls | Spawn isolated 1-message conversations; not part of the main thread |
| After `/compact` | History resets to 1 message (compacted summary replaces full history) |

**Implication for prefix caching:** In this simple setting, the prefix `msg_0 … msg_{N-2}` is completely stable between turns, so a KV cache server can reuse cached states for all but the last 1–2 messages. This is the maximally cache-friendly baseline, not the full story for Claude Code overall.

## Routing Strategy

Start with this path:

```text
Claude Code
  -> http://127.0.0.1:8080/v1/messages
Logging Proxy
  -> https://api.anthropic.com/v1/messages
Anthropic API
```

Optional later phase:

```text
Claude Code
  -> Your Logging Proxy
  -> vLLM Server
```

## Claude Code Configuration

Configure Claude Code through `~/.claude/settings.json`
originally:
```json
{
  "model": "sonnet"
}
```

Recommended initial configuration:

```json
{
  "model": "sonnet",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8201",
    "ANTHROPIC_API_KEY": "ACTUAL_ANTHROPIC_API_KEY",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-sonnet-4-6"
  }
}
```

## Trace Output

Write append-only JSONL files under:

```text
traces/<session_name>_trace.jsonl
```

## Trace Collection Procedure

**Step 1.** Spin up the proxy:

```bash
uv run python anthropic_proxy.py
```

**Step 2.** Check if the endpoint is reachable:

```bash
curl https://api.anthropic.com/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Hi"}
    ]
  }'
```

**Step 3.** In another terminal, start a session:

```bash
curl -X POST http://localhost:8201/session/start \
  -H "content-type: application/json" \
  -d '{"name": "testing_session"}'
```

**Step 4.** In another terminal, ask questions

## testing_session traces

Questions asked in claude code:

1. I want you to help me analyze this repository. Please use your tools to search for and read the full source code of the core caching engine, the memory allocator, and the server logic. Read at least 5-7 core large python files in full so you have the complete context of how LMCache is working under the hood. Once you have read them, please give me a brief summary.

For that first question, Claude Code did not read 5–7 large files in full. Instead, it:
explored the repo structure with ls, find, and wc -l, identified candidate core files, then sampled many files with bounded Read calls
read 15 files but the important nuance is: these were mostly partial reads, not full-file reads.

The actual Read limits were typically:80/60/50 lines. So the model mostly did is file discovery, line-count ranking, targeted partial reads of many candidate files but not "read 5-7 large Python files end to end".

2. In your earlier repo analysis, you mentioned the token hashing granularity. What exact chunk size did you say LMCache uses, and why does that matter for reuse?
3. `/compact` command
4. Earlier you said the system keeps the five most recently accessed files or, if that was your own inference, please verify the exact file(s) and code path that support your claim about lookup/prefetch behavior. Quote the function names and explain the call chain.
5. `/compact` command
6. Without starting over, what important details from our earlier discussion are you still carrying forward right now, and which details are likely compressed away?

> In total: 6 user queries and 106 raw records in 1063.297s for `testing_session` (including user thinking time). The 106 records break down into 53 requests and 53 responses. Inside the 53 requests, 30 are `POST /v1/messages`, 22 are `POST /v1/messages/count_tokens`, and 1 is `HEAD /` (health/reachability check). Claude Code also does not behave like a single linear thread: it has a parent thread plus child LLM calls, and the parent consumes child outputs without necessarily carrying the children's full internal history.


### Traces Interpretation
`testing_session` is not pure prefix append end-to-end, but it is still mostly prefix-friendly. The main non-prefix behavior comes from `/compact`, which creates append-with-reset segments rather than frequent mid-prompt rewrites.

Takeaway:

- This session is useful for showing that Claude Code can leave the pure append regime.
- However, the dominant pattern is still prefix growth within a segment.
- `/compact` is therefore a reliable trigger for collecting non-prefix traces, but it is still a coarse mechanism.
- For richer non-prefix behavior, the next step is to collect workloads where the harness itself rewrites context, such as rolling summaries, retrieval refresh, or tool-state digests.

## re-read after edit testing
Use a small synthetic evaluation built on top of real LMCache PRs. The goal is to test whether Claude Code relies on prior memory, reads only the diff, or re-reads the full file after a small edit.

Recommended turn structure:

1. Ask Claude Code to read a large file in full and explain the relevant code path.
2. Introduce a PR with a small but meaningful change.
3. Ask how that change affects the surrounding behavior.
4. Ask for re-validation against the full file, not just the diff.

Questions asked in claude code:
- Turn 1:
Read the full `lmcache/integration/vllm/vllm_v1_adapter.py` file end to end and explain how `build_connector_meta` fits into the decode/save_decode_cache flow. Cite 5 function names from different parts of the file.

- Turn 2:
Could you please help to inspect PR #2929: https://github.com/LMCache/LMCache/pull/2929
Explain how the changed line(s) affect decoding behavior and preemption handling.

- Turn 3:
Do not rely only on earlier memory or on the diff. Re-read the full current `vllm_v1_adapter.py` and verify whether your Turn 2 explanation still holds in the context of the whole file. Give one supporting untouched code location and one potentially conflicting untouched code location.

- Turn 4:
If a reviewer had only read the diff, what would they likely miss compared with re-reading the full file?

> In total: 4 user queries and 50 raw records in 550.841s for `testing_reRead_session` (including user thinking time). The turn structure is: Turn 1 forces a full-file understanding pass; Turn 2 introduces the PR and observes whether Claude Code answers from memory, diff-local context, or fresh reads; Turn 3 explicitly forces re-validation against the full file; Turn 4 checks what re-reading added over diff-only review.

### Traces Interpretation
This workflow is useful for observing whether Claude Code relies on memory, diff-local reasoning, or fresh file reads after a small code change.

What the trace shows:

- Turn 1 builds a large file-level context, but the file is read progressively because it is too large for a single read.
- Turn 2 adds PR and diff context on top of that existing history.
- Turn 3 triggers fresh reads from additional untouched regions of the file.
- Even with those fresh reads, the parent thread remains mostly prefix-friendly: Claude Code keeps prior context and appends new reads and reasoning.

Takeaway:

- `A + turn2 + fresh_reads` is a useful first-step workload for studying re-read behavior.
- It is still mostly prefix-append, so it is not the strongest non-prefix or CacheBlend-style case by itself.
- A stronger target would compress or replace earlier raw context, for example `summary(A) + turn2 + refreshed_chunks(A')`.

# Side Notes
Claude Code does not guarantee non-prefix behavior even without `/compact`:
1. Claude Code switches to a different internal persona/sub-agent thread
2. the harness rewrites system or context blocks
3. retrieved context is refreshed instead of preserved verbatim
4. some internal summarization or memory management kicks in