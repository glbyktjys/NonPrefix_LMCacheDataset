# Trace Conclusions

This folder contains two converted Claude Code trace sessions:
- `testing_compact_session_converted.jsonl`: 30 usable trace entries
- `testing_reRead_session_trace_converted.jsonl`: 24 usable trace entries

## What We Learned

- Claude Code is mostly prefix-friendly within a thread or segment.
- `/compact` is the clearest trigger we observed for non-prefix behavior; it creates append-with-reset segments.
- In reread-after-edit testing, Claude Code usually did not replace old file context with a fresh full-file snapshot.
- Instead, it kept prior history and appended diff context, fresh local reads, and new reasoning.

## Repo-Analysis Session

- For the first large repo-analysis question, Claude Code touched 15 distinct files.
- Those reads were mostly partial, not full-file rereads.
- Peak effective prompt size during that phase was about 45.7k tokens, so the model used substantial context but did not fill the whole context window.

## How We Measured This

- Distinct files read: counted from `Read` tool calls in the converted trace.
- Effective prompt tokens: `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
- This uses Anthropic response `usage` fields as a practical estimate of total prompt footprint.
- Cached tokens still belong to the logical context; caching reuses computation, but does not remove those tokens from the effective prompt.

## Takeaway

- Current traces suggest Claude Code usually updates context by appending new evidence rather than rewriting old raw context in place.
- For stronger non-prefix traces, the best next step is rolling summaries or other context-refresh mechanisms.
