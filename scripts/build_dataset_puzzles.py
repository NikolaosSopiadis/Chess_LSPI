# scripts/build_dataset_puzzles.py
from __future__ import annotations

import argparse
from pathlib import Path

from chess_rl.data.build_samples_puzzles import build_samples_from_puzzles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True, help="Path to lichess puzzles CSV (plain .csv)")
    ap.add_argument("--out", required=True, help="Output samples.jsonl.gz")
    ap.add_argument("--features", default="v1_basic", help="Feature registry key (e.g. v1_basic)")
    ap.add_argument("--min-rating", type=int, default=1800)
    ap.add_argument("--max-rows", type=int, default=50_000)
    ap.add_argument("--themes", default="", help="Comma-separated themes filter (optional)")

    ap.add_argument("--workers", type=int, default=0, help="0 => auto (cpu_count-1), 1 => no parallel")
    ap.add_argument("--chunksize", type=int, default=32, help="Task chunk size for multiprocessing")
    ap.add_argument("--reward-alpha", type=float, default=0.05)

    args = ap.parse_args()

    include_themes = None
    if args.themes.strip():
        include_themes = {t.strip() for t in args.themes.split(",") if t.strip()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    build_samples_from_puzzles(
        puzzles_csv=args.puzzles,
        out_jsonl_gz=str(out),
        feature_name=args.features,
        reward_alpha=args.reward_alpha,
        include_themes=include_themes,
        min_rating=args.min_rating,
        max_rows=args.max_rows,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
