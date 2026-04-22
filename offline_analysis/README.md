# Offline Analysis

Offline CacheBlend-style reuse analysis for Anthropic-format and OpenAI-format raw traces.

Estimates prefix reusable, non-prefix reusable, and new compute tokens per session, and generates an interactive HTML dashboard to visualize chunk reuse.

The estimator is inspired by CacheBlend, but it is still an offline approximation. It does not run vLLM or LMCache, and it does not model Blend recompute exactly.

## Usage

The analyzer supports two trace formats via `--format`:

- **`claudecode`** (default) — Anthropic API traces (`/v1/messages`). Prompt is serialized as: tools, system, messages.
- **`openclaw`** — OpenAI/vLLM traces (`/v1/chat/completions`). System prompt is `messages[0]` with `role="system"`. Tools use OpenAI function-call format.

### Anthropic format (Claude Code traces)

```bash
python offline_analysis/analyze_trace.py \
  raw_traces/ClaudeCode/testing_compact_session_trace.jsonl
```

### OpenAI format (OpenClaw / vLLM traces)

```bash
python offline_analysis/analyze_trace.py \
  non_prefix_raw_traces/mtRag/clapnq_clapnq_0208bf26ec357a803445290fa88a2e9e_trace.jsonl \
  --format openclaw
```

### HTML dashboard

```bash
python offline_analysis/analyze_trace.py \
  raw_traces/ClaudeCode/testing_compact_session_trace.jsonl \
  --html output.html
```

Then open `output.html` in your browser. Click a segment in the bar to follow where that text appears in later turns. Shift+click to track multiple regions.

### Other options

```bash
python offline_analysis/analyze_trace.py \
  raw_traces/ClaudeCode/testing_compact_session_trace.jsonl \
  --output my_analysis.json \
  --html my_dashboard.html \
  --chunk-size 256 \
  --log-each-entry \
  --include-assistant-thinking
```

## Files

- `cacheblend_hashes.py` — core two-hash engine: rolling prefix storage hashes + polynomial content fingerprints
- `analyze_trace.py` — reads Anthropic or OpenAI/OpenClaw raw traces (`--format claudecode|openclaw`), reconstructs prompts, computes reuse metrics, generates analysis JSON and optional HTML dashboard
- `trace_viewer.html` — HTML template for the interactive dashboard (used by `analyze_trace.py --html`)

## Goal

Given a raw JSONL trace, group requests by session and estimate:

- `input_tokens`
- `prefix_tokens`
- `nonprefix_reusable_tokens`
- `new_compute_tokens`

Two cache-source modes (both computed automatically):

- `request_only` — only prior `/v1/messages` requests populate the cache
- `include_decode_cache` — assistant responses also populate the cache (models `save_decode_cache=true`)

## What Gets Counted

The analyzer only scores real inference requests:

- **claudecode format**: `type="request"` with `path="/v1/messages"`
- **openclaw format**: `type="request"` with `path="/v1/chat/completions"`

It does not score:

- `type="request"` with `path="/v1/messages/count_tokens"`
- outer trace metadata, headers, or transport wrapper JSON

Responses are used for:

- provider `usage` validation
- optional decode-cache insertion in `include_decode_cache` mode

## Two Hashes

The estimator uses two separate hashes, mirroring the CacheBlend design:

### 1. Rolling Prefix Storage Hash

Position-dependent, acts like the storage key.

```text
chunks: [AB] [CD] [EF]

rolling hashes:
  h(AB)
  h(ABCD)
  h(ABCDEF)
```

Implemented in `RollingPrefixHasher` in `cacheblend_hashes.py`.

### 2. Content-Only Fingerprint

Position-independent, used to find reusable chunk content anywhere in a later request.

```text
chunks: [AB] [CD] [EF]

fingerprints:
  fp(AB)
  fp(CD)
  fp(EF)
```

Implemented in `_poly_hash`, `rolling_window_fingerprints`, and `BlendTokenRangeMatcher` in `cacheblend_hashes.py`.

## How We Analyze One Request

For each inference request in a session:

1. Build the model-visible prompt:
   - **claudecode**: from `body.system`, `body.messages`, `body.tools`; serialized as tools → system → messages
   - **openclaw**: from `body.messages` (system is `messages[0]`), `body.tools`; serialized as messages (system-first)
3. Tokenize with `tiktoken o200k_base`
4. Compute rolling storage hashes
5. Compare against prior request sequences for prefix reuse
6. Run the non-prefix matcher for reusable chunk content after the prefix span
7. Keep only non-overlapping matches
8. Count the rest as new compute
9. Add current request into both caches
10. In `include_decode_cache` mode, also add the assistant response

## Metrics

Per session:

| Metric | Description |
|---|---|
| `input_tokens` | total input tokens across all requests |
| `prefix_tokens` | tokens reused via prefix cache |
| `nonprefix_reusable_tokens` | tokens reused via non-prefix content matching |
| `new_compute_tokens` | tokens requiring fresh computation |

Ratios are normalized by `input_tokens`:

```text
prefix_ratio             = prefix_tokens / input_tokens
nonprefix_reuse_ratio    = nonprefix_reusable_tokens / input_tokens
new_compute_ratio        = new_compute_tokens / input_tokens
```

## Note

This is still an offline estimator:

- it uses `tiktoken o200k_base`, not the real Anthropic tokenizer
- it does not model runtime cache eviction
- it does not explicitly model Blend recompute cost
- it estimates reusable chunks, not exact runtime latency savings
