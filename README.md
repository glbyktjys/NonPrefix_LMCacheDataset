# Claude Code Logging Proxy Plan

## Goal

Route local Claude Code through a custom proxy, log all Anthropic API requests and responses, and analyze the traces to understand how Claude Code carries conversation history across turns.

The main question is whether Claude Code:
- mostly appends prior history as a stable prefix, or
- dynamically injects, rewrites, trims, or summarizes earlier context

This matters for prefix reuse:
- append-like behavior is more prefix-cache-friendly
- middle-of-prompt mutation reduces reusable prefix and motivates approaches like CacheBlend

### Finding (pre-`/compact`)

**Claude Code uses pure prefix append with a sliding prompt-cache marker.**

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

**Implication for prefix caching:** Because the prefix `msg_0 … msg_{N-2}` is completely stable between turns, a KV cache server can reuse cached states for all but the last 1–2 messages. This is maximally cache-friendly — no CacheBlend-style mid-prompt patching is needed for the base case.

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

{
  "model": "sonnet",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8201",
    "ANTHROPIC_API_KEY":"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",
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
2. In your earlier repo analysis, you mentioned the token hashing granularity. What exact chunk size did you say LMCache uses, and why does that matter for reuse?
3. `/compact` command
4. Earlier you said the system keeps the five most recently accessed files or, if that was your own inference, please verify the exact file(s) and code path that support your claim about lookup/prefetch behavior. Quote the function names and explain the call chain.
5. `/compact` command
6. Without starting over, what important details from our earlier discussion are you still carrying forward right now, and which details are likely compressed away?

> In total: 6 user queries, 106 LLM requests in 1063.297s for `testing_session` (includes user thinking time between questions). In 106 total records written, there are 53 request records and 53 response records. Inside 53 requests, 30 are POST /v1/messages, 22 are POST /v1/messages/count_tokens, and 1 is HEAD/(health/reachability check). And one more important things to notes, claude code would have a parent thread and have several child LLM calls and each chilren produce outputs and parent consumes the outputs, not necessarily the children's full internal history.


### Traces Interpretation
testing_session` is not pure prefix append end-to-end, but it is still mostly prefix-friendly. So this trace shows that we can manually create non-prefix behavior, but the dominant pattern is still append-with-reset. If the goal is to study realistic non-prefix reuse, we should collect workloads where the harness itself regularly rewrites context.

Natural non-prefix patterns to target:

1. `rolling summaries`
   Replace older dialogue with a refreshed summary every few turns.
   Example:
   - Turn 1-5: keep full history
   - Turn 6: replace turns 1-4 with "Summary so far: user wants X, code path Y was inspected, bug likely in Z"
   This breaks the old prefix even though the task is still the same.

2. `retrieval refresh`
   Instead of replaying the whole conversation, re-insert only the currently relevant memory or files.
   Example:
   - Early turns retrieve `storage_manager.py` and `local_cpu_backend.py`
   - Later turns drop those and inject a new short memory pack about `token_database.py` and chunk size
   The prompt keeps the same task, but the middle context is replaced by a different retrieved set.

3. `changing tool state digests`
   Recompute a compact state summary from tool results and send that digest instead of the raw prior outputs.
   Example:
   - After several shell reads, replace raw file dumps with:
     "Workspace state: inspected 7 files; confirmed chunk_size=256; no evidence for fixed 5-file prefetch policy"
   This mutates earlier tool context into a shorter synthesized state block.

4. `dynamic user/agent interaction`
   Multi-turn tasks where new user information changes which prior details matter.
   Example:
   - User first asks for root-cause analysis
   - Later says "ignore performance, focus only on correctness" or "now explain this to a new engineer"
   The agent may keep only the facts relevant to the new goal and rewrite or compress the rest.

Practical takeaway:

- `append-only` traces are good for measuring prefix reuse in the best case.
- `append-with-reset` traces from `/compact` are useful, but still coarse.
- The most interesting non-prefix traces come from summary refresh, retrieval refresh, and state-digest updates within the same ongoing task.

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

> In total: 4 user queries, 50 LLM requests in 550.841s for `testing_reRead_session` (includes user thinking time between questions). Essentially in these 4 turns: Turn 1: force a full-file understanding pass; Turn 2(observation turn): introduce the PR and observe whether Claude Code can answer from memory, local diff context, or by re-reading more; Turn 3: explicitly force a fresh full-file re-validation; Turn 4: verify what re-reading added over diff-only review

# Side Notes
Claude Code doesn't guaranteed to provide non-prefix behavior even without /compact:
1. Claude Code switches to a different internal persona/sub-agent thread
2. the harness rewrites system/context blocks
3. retrieved context is refreshed instead of preserved verbatim
4. some internal summarization/memory management kicks in