"""
Render MT-RAG/OpenClaw trace visualizations as self-contained HTML files.

Each row corresponds to one normal backend request turn, with metrics computed
against the previous normal turn:
    - total tokens (approximate)
    - prefix tokens
    - non-prefix reused tokens
    - compaction event flag
    - compacted state flag
    - prefix-break flag

The stacked bar inside the "Total Tokens" cell uses:
    - orange: direct prefix reuse
    - green: reused non-prefix content
    - black: everything else
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from analyze_compaction import (
    _has_inserted_summary,
    _load_requests,
    _longest_suffix_reused_in_current,
    _request_token_estimate,
    _tail_token_estimate,
)
from proxy.rollingHashes import build_sequence_state, compare_sequence_states


def _build_turn_rows(path: Path) -> list[dict[str, object]]:
    requests = _load_requests(path)
    rows: list[dict[str, object]] = []

    previous_normal = None
    previous_state = None
    pending_compact_requests = []

    for request in requests:
        if request.compact_request:
            pending_compact_requests.append(request)
            continue

        current_state = build_sequence_state(request.messages)
        total_tokens = _request_token_estimate(request.messages)
        compaction_event_flag = bool(pending_compact_requests)
        compacted_state_flag = _has_inserted_summary(request.messages)

        if previous_normal is None or previous_state is None:
            prefix_tokens = 0
            non_prefix_tokens = 0
            prefix_break_flag = False
        else:
            comparison = compare_sequence_states(current_state, previous_state)
            prefix_tokens = round(comparison["shared_prefix_chunk_ratio"] * total_tokens)
            remaining_tokens = max(total_tokens - prefix_tokens, 0)
            non_prefix_tokens = round(
                comparison["reused_nonprefix_chunk_ratio"] * remaining_tokens
            )

            if compacted_state_flag:
                retained_tail_messages, _ = _longest_suffix_reused_in_current(
                    previous_normal.messages,
                    request.messages,
                )
                retained_tail_tokens = _tail_token_estimate(
                    previous_normal.messages,
                    retained_tail_messages,
                )
                if retained_tail_tokens:
                    non_prefix_tokens = min(remaining_tokens, retained_tail_tokens)

            non_prefix_tokens = max(0, min(non_prefix_tokens, remaining_tokens))
            prefix_break_flag = (
                previous_normal is not None
                and comparison["shared_prefix_chunk_ratio"] == 0.0
                and (compaction_event_flag or comparison["reused_nonprefix_chunk_count"] > 0)
            )

        rows.append(
            {
                "turn_id": len(rows) + 1,
                "total_tokens": total_tokens,
                "prefix_tokens": prefix_tokens,
                "non_prefix_tokens": non_prefix_tokens,
                "other_tokens": max(total_tokens - prefix_tokens - non_prefix_tokens, 0),
                "compaction_event_flag": compaction_event_flag,
                "compacted_state_flag": compacted_state_flag,
                "prefix_break_flag": prefix_break_flag,
                "request_id": request.request_id,
                "raw_request_index": request.index,
                "compact_request_indexes": [
                    compact_request.index for compact_request in pending_compact_requests
                ],
            }
        )

        previous_normal = request
        previous_state = current_state
        pending_compact_requests = []

    return rows


def _bar_segments(row: dict[str, object]) -> tuple[float, float, float]:
    total = int(row["total_tokens"])
    if total <= 0:
        return 0.0, 0.0, 0.0
    prefix = 100.0 * int(row["prefix_tokens"]) / total
    non_prefix = 100.0 * int(row["non_prefix_tokens"]) / total
    other = max(0.0, 100.0 - prefix - non_prefix)
    return prefix, non_prefix, other


def _format_int(value: object) -> str:
    return f"{int(value):,}"


def _render_html(trace_path: Path, rows: list[dict[str, object]]) -> str:
    table_rows: list[str] = []
    for row in rows:
        prefix_pct, non_prefix_pct, other_pct = _bar_segments(row)
        tooltip = (
            f"request_id={row['request_id']} | raw_request_index={row['raw_request_index']}"
        )
        if row["compact_request_indexes"]:
            tooltip += f" | compact_requests={row['compact_request_indexes']}"

        table_rows.append(
            f"""
            <tr title="{html.escape(tooltip)}">
              <td>{row['turn_id']}</td>
              <td>
                <div class="token-cell">
                  <div class="token-value">{_format_int(row['total_tokens'])}</div>
                  <div class="bar">
                    <span class="seg prefix" style="width:{prefix_pct:.4f}%"></span>
                    <span class="seg nonprefix" style="width:{non_prefix_pct:.4f}%"></span>
                    <span class="seg other" style="width:{other_pct:.4f}%"></span>
                  </div>
                </div>
              </td>
              <td>{_format_int(row['prefix_tokens'])}</td>
              <td>{_format_int(row['non_prefix_tokens'])}</td>
              <td><span class="flag {'yes' if row['compaction_event_flag'] else 'no'}">{'yes' if row['compaction_event_flag'] else 'no'}</span></td>
              <td><span class="flag {'yes' if row['compacted_state_flag'] else 'no'}">{'yes' if row['compacted_state_flag'] else 'no'}</span></td>
              <td><span class="flag {'yes' if row['prefix_break_flag'] else 'no'}">{'yes' if row['prefix_break_flag'] else 'no'}</span></td>
            </tr>
            """.strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(trace_path.name)} trace view</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: #fffaf2;
      --ink: #1e1a17;
      --grid: #d8cdbf;
      --prefix: #d97706;
      --nonprefix: #2f855a;
      --other: #111111;
      --yes: #0f766e;
      --no: #7c2d12;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background:
        radial-gradient(circle at top left, #efe2c8 0, transparent 30%),
        linear-gradient(180deg, #f6f3eb 0%, var(--bg) 100%);
      color: var(--ink);
      font: 14px/1.4 Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .meta {{
      margin: 0 0 18px;
      color: #5b5249;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin: 0 0 18px;
      padding: 12px 14px;
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--grid);
      border-radius: 12px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.2);
    }}
    .table-wrap {{
      overflow-x: auto;
      background: rgba(255, 250, 242, 0.94);
      border: 1px solid var(--grid);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(31, 22, 14, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #efe5d6;
      text-align: left;
      padding: 12px 14px;
      border-bottom: 1px solid var(--grid);
      white-space: nowrap;
    }}
    tbody td {{
      padding: 10px 14px;
      border-bottom: 1px solid rgba(216, 205, 191, 0.8);
      vertical-align: middle;
      white-space: nowrap;
    }}
    tbody tr:nth-child(even) {{
      background: rgba(239, 229, 214, 0.28);
    }}
    .token-cell {{
      min-width: 310px;
    }}
    .token-value {{
      margin-bottom: 6px;
    }}
    .bar {{
      display: flex;
      width: 100%;
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: #d9d1c7;
    }}
    .seg {{
      height: 100%;
    }}
    .prefix {{ background: var(--prefix); }}
    .nonprefix {{ background: var(--nonprefix); }}
    .other {{ background: var(--other); }}
    .flag {{
      display: inline-block;
      min-width: 38px;
      text-align: center;
      padding: 3px 8px;
      border-radius: 999px;
      color: white;
      font-size: 12px;
      letter-spacing: 0.02em;
      text-transform: lowercase;
    }}
    .flag.yes {{ background: var(--yes); }}
    .flag.no {{ background: var(--no); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(trace_path.name)}</h1>
    <p class="meta">Per-turn view based on the previous normal turn. Token counts are approximate because the raw traces do not include model prompt usage. "Compaction event" means internal compact requests happened immediately before this visible turn. "Compacted state" means the visible turn still starts from an inserted summary.</p>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:var(--prefix)"></span> prefix</span>
      <span class="legend-item"><span class="swatch" style="background:var(--nonprefix)"></span> non-prefix reused</span>
      <span class="legend-item"><span class="swatch" style="background:var(--other)"></span> other</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Turn ID</th>
            <th>Total Tokens</th>
            <th>Prefix Tokens</th>
            <th>Non-Prefix Tokens</th>
            <th>Compaction Event</th>
            <th>Compacted State</th>
            <th>Prefix-Break Flag</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def _render_index(index_rows: list[tuple[str, str]]) -> str:
    links = "\n".join(
        f'<li><a href="{html.escape(rel_path)}">{html.escape(label)}</a></li>'
        for label, rel_path in index_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trace Visualizations</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: #f6f3eb;
      color: #201a17;
      font: 16px/1.5 Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    .panel {{
      max-width: 900px;
      margin: 0 auto;
      padding: 24px;
      background: #fffaf2;
      border: 1px solid #d8cdbf;
      border-radius: 16px;
    }}
    h1 {{ margin-top: 0; }}
    a {{ color: #9a3412; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Trace Visualizations</h1>
    <ul>
      {links}
    </ul>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=sorted((Path(__file__).resolve().parent / "traces").glob("*.jsonl")),
        help="Trace JSONL files to visualize. Defaults to raw_traces/mtRag/traces/*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "visualizations",
        help="Directory where HTML files should be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[tuple[str, str]] = []

    for path in args.paths:
        rows = _build_turn_rows(path.resolve())
        output_path = args.output_dir / f"{path.stem}.html"
        output_path.write_text(_render_html(path, rows), encoding="utf-8")
        index_rows.append((path.name, output_path.name))
        print(f"{path.name} -> {output_path}")

    index_path = args.output_dir / "index.html"
    index_path.write_text(_render_index(index_rows), encoding="utf-8")
    print(f"index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
