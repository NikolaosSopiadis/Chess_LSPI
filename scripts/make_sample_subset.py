# scripts/make_sample_subset.py
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import TextIO

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


def open_text(path: str | Path, mode: str) -> TextIO:
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8")


def reservoir_sample(src: str, n: int, seed: int, *, validate: bool) -> tuple[list[str], int]:
    rng = random.Random(seed)
    reservoir: list[str] = []

    with open_text(src, "rt") as f:
        it = f
        if tqdm is not None:
            it = tqdm(f, desc="sampling", unit="lines", mininterval=0.25)

        total = 0

        for line_no, line in enumerate(it, start=1):
            line = line.strip()
            if not line:
                continue

            if validate:
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at input line {line_no}: {e}") from e

            i = total
            total += 1

            if i < n:
                reservoir.append(line + "\n")
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line + "\n"

    return reservoir, total


def first_n(src: str, n: int, *, validate: bool) -> tuple[list[str], int]:
    rows: list[str] = []
    total = 0

    with open_text(src, "rt") as f:
        it = f
        if tqdm is not None:
            it = tqdm(f, desc="reading first N", unit="lines", mininterval=0.25)

        for line_no, line in enumerate(it, start=1):
            line = line.strip()
            if not line:
                continue

            if validate:
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at input line {line_no}: {e}") from e

            total += 1

            if len(rows) < n:
                rows.append(line + "\n")
            else:
                break

    return rows, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument(
        "--mode",
        choices=["reservoir", "first-n"],
        default="reservoir",
        help="reservoir gives a random sample over the whole file; first-n just takes the first N rows.",
    )
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip JSON validation for speed.",
    )
    args = ap.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    validate = not args.no_validate

    if args.mode == "reservoir":
        rows, total_seen = reservoir_sample(args.src, args.samples, args.seed, validate=validate)
    else:
        rows, total_seen = first_n(args.src, args.samples, validate=validate)

    if len(rows) < args.samples:
        print(
            f"Warning: requested {args.samples} samples, "
            f"but source only had {len(rows)} usable rows."
        )

    if args.shuffle:
        rng = random.Random(args.seed + 999)
        rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open_text(out, "wt") as f:
        it = rows
        if tqdm is not None:
            it = tqdm(rows, desc="writing", unit="lines", mininterval=0.25)

        for line in it:
            f.write(line)

    print(f"Wrote {len(rows)} samples to {out}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Input usable rows seen: {total_seen}")


if __name__ == "__main__":
    main()