# OpenClaw Resume Trace Notes

```text
OpenClaw (VPS:18789)
  → configured upstream: http://127.0.0.1:18790   ← your proxy
    → BACKEND_URL=http://<vllm-server-ip>:8200     ← vLLM
```

This folder contains `openclaw_resume_test_trace.jsonl`, which has `48` raw records total:
- `24` request records
- `24` response records

Using raw-record numbering in the trace:
- records `0-3` are Session 1 (Japan trip planning, first 2 turns)
- records `4-41` are Session 2 (rubber mats, long tool-heavy search loop)
- records `42-47` are the return to Session 1

## Main takeaway

OpenClaw looks **session-local prefix-friendly** in this trace.
- Session 1 keeps its own history and resumes cleanly when we come back to it.
- Session 2 also keeps its own history as a separate append-only chain.
- So the resumed Session 1 prompt does **not** look like it was rebuilt from scratch or corrupted.

The practical cache problem is more subtle:
- if we compare Session 1 to its own earlier state, prefix reuse is good
- but if we compare globally adjacent turns across interleaved sessions, overlap is very small
- for example, using raw-record numbering, the resumed Session 1 request at record `42` overlaps strongly with its true earlier Session 1 request at record `2`, but has almost no useful overlap with the immediately previous Session 2 request at record `40`
- in request-level terms, this is the same pattern as `lcp(req2, req22) = 11` versus `lcp(req21, req22) = 2`

This means session interleaving is bad for **naive global prefix caching**:
- a cache system that mainly benefits from the immediately previous request will miss reuse here
- in this trace, comparing raw record `40` to raw record `42` gives almost no reusable overlap, even though raw record `42` still cleanly continues Session 1
- a session-aware or chain-aware cache can still recover the right prefix

## Tool usage remarks

Tool results are part of the conversation history and get appended into later requests.
- This makes Session 2 grow very quickly even though the user only asked 2 questions.
- Much of the growth comes from repeated assistant/tool loops, not new user information.
- So even when prefix reuse is clean inside one session, tool-heavy queries can still create very large contexts and reduce practical cache efficiency.

# OpenClaw natural context compaction testing case
OpenClaw’s docs(https://docs.openclaw.ai/concepts/compaction) describe compaction as: older turns are summarized into a compact entry, that summary is saved in the transcript, and recent messages are kept intact so it's very similar to our proposed case:

    turn 1: A B C D E F G
    compact older prefix -> S1
    next visible context: S1 + E F G

    turn 2: S1 E F G H I
    compact older prefix again -> S2
    next visible context: S2 + G H I

## start to record session traces
curl -X POST http://localhost:8201/session/start \
  -H "content-type: application/json" \
  -d '{"name": "openclaw_compact_test"}'

- Turn 1: I have uploaded Tesla's four years of 10K and could you please read the 2022 Tesla 10-K and summarize the key business model, risks, and growth priorities.
- Turn 2: Read the 2023 Tesla 10-K and compare it with 2022. Keep only material changes.
- Turn 3: Create a concise 2022–2023 thesis for Tesla covering demand, pricing, autonomy, manufacturing, energy, and regulation.
- Turn 4: Read the 2024 Tesla 10-K and revise the thesis. Explicitly drop any claims that no longer hold.
- Turn 5: List the top 5 narrative shifts between 2022 and 2024.
- Turn 6: Read the 2025 Tesla 10-K and update the thesis again.
- Turn 7: Produce a 4-year evolution summary with only durable conclusions and unresolved questions.
- Turn 8: Now give the final ranking of the three most important long-term Tesla drivers supported by the filings.

Far from reaching the context window limits -> trying use mtRag instead see: raw_traces/mtRag_on_OpenClaw