from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Iterator, Any

from tqdm.auto import tqdm

from chess_core.board import Board
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--features", required=True)
    ap.add_argument("--max-samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=4.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Reservoir sample source rows so the anchor is spread across the PGN file.
    reservoir: list[dict[str, Any]] = []

    seen = 0
    for rec in tqdm(iter_jsonl_gz(args.src), desc="read source", unit="rows"):
        if rec.get("feature_version") != args.features:
            raise ValueError(
                f"feature mismatch: rec={rec.get('feature_version')!r}, "
                f"requested={args.features!r}"
            )

        seen += 1

        if len(reservoir) < args.max_samples:
            reservoir.append(rec)
        else:
            j = rng.randrange(seen)
            if j < args.max_samples:
                reservoir[j] = rec

    rng.shuffle(reservoir)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    b = Board()
    written = 0

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for rec in tqdm(reservoir, desc="write anchors", unit="rows"):
            fen_next = str(rec["fen_next"])
            b.init_board(fen_next)

            value = float(material_potential(b)) * args.scale

            if args.clip > 0:
                value = max(-args.clip, min(args.clip, value))

            out_rec = {
                "feature_version": args.features,
                "reward_version": "material_value_anchor_v1",
                "phi": rec["phi"],
                "r": value,
                "done": True,
                "fen_next": fen_next,
                "source": "material_anchor_from_pgn",
            }

            f.write(json.dumps(out_rec, separators=(",", ":")) + "\n")
            written += 1

    print(f"wrote: {args.out}")
    print(f"source rows seen: {seen}")
    print(f"anchors written: {written}")
    print(f"scale: {args.scale}")
    print(f"clip: ±{args.clip}")


if __name__ == "__main__":
    main()