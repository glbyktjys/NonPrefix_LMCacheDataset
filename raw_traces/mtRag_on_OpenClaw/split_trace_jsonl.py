"""
Split large JSONL trace files into smaller chunks without breaking lines.

Default output layout:
    raw_traces/mtRag/splits/<trace-stem>/<trace-stem>.part001.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MAX_MB = 50


def _chunk_path(output_dir: Path, stem: str, part_index: int) -> Path:
    return output_dir / f"{stem}.part{part_index:03d}.jsonl"


def split_jsonl(path: Path, output_root: Path, max_bytes: int) -> list[Path]:
    output_dir = output_root / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    part_index = 0
    current_path: Path | None = None
    current_handle = None
    current_bytes = 0

    try:
        with path.open("rb") as source:
            for line in source:
                if not line.strip():
                    continue

                if current_handle is None or (current_bytes and current_bytes + len(line) > max_bytes):
                    if current_handle is not None:
                        current_handle.close()
                    part_index += 1
                    current_path = _chunk_path(output_dir, path.stem, part_index)
                    current_handle = current_path.open("wb")
                    written_paths.append(current_path)
                    current_bytes = 0

                current_handle.write(line)
                current_bytes += len(line)
    finally:
        if current_handle is not None:
            current_handle.close()

    return written_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=sorted((Path(__file__).resolve().parent / "traces").glob("*.jsonl")),
        help="Trace JSONL files to split. Defaults to raw_traces/mtRag/traces/*.jsonl",
    )
    parser.add_argument(
        "--max-mb",
        type=int,
        default=DEFAULT_MAX_MB,
        help=f"Maximum chunk size in megabytes. Default: {DEFAULT_MAX_MB}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "splits",
        help="Directory where split trace folders should be written.",
    )
    args = parser.parse_args()

    max_bytes = args.max_mb * 1024 * 1024
    for path in args.paths:
        written = split_jsonl(path.resolve(), args.output_root.resolve(), max_bytes)
        print(f"{path}: {len(written)} parts")
        for part in written:
            print(f"  {part} ({part.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
