from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import TextIO

from tqdm import tqdm


def open_text(path: str | Path, mode: str) -> TextIO:
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8")


def parse_src(spec: str) -> tuple[str, int]:
    # path:count
    # Use rsplit so paths containing ":" are less problematic.
    path, count_s = spec.rsplit(":", 1)
    return path, int(count_s)


def reservoir_sample(path: str, n: int, seed: int, *, validate: bool = True) -> tuple[list[str], int]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    total = 0

    with open_text(path, "rt") as f:
        for line_no, line in enumerate(tqdm(f, desc=f"sampling {Path(path).name}", unit="lines"), start=1):
            line = line.strip()
            if not line:
                continue

            if validate:
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e

            i = total
            total += 1

            if i < n:
                reservoir.append(line + "\n")
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line + "\n"

    return reservoir, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        action="append",
        required=True,
        help="Source dataset and quota as path:count. May be repeated.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    validate = not args.no_validate

    all_rows: list[str] = []
    manifest: list[dict] = []

    for i, src_spec in enumerate(args.src):
        path, count = parse_src(src_spec)

        rows, total_seen = reservoir_sample(
            path,
            count,
            args.seed + 1009 * i,
            validate=validate,
        )

        if len(rows) < count:
            print(f"Warning: requested {count} rows from {path}, but only got {len(rows)}.")

        all_rows.extend(rows)
        manifest.append(
            {
                "path": path,
                "requested": count,
                "written": len(rows),
                "source_rows_seen": total_seen,
            }
        )

    rng = random.Random(args.seed + 999_999)
    rng.shuffle(all_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open_text(out, "wt") as f:
        for line in tqdm(all_rows, desc="writing mix", unit="rows"):
            f.write(line)

    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    with open(manifest_path, "wt", encoding="utf-8") as f:
        json.dump(
            {
                "out": str(out),
                "seed": args.seed,
                "total_rows": len(all_rows),
                "sources": manifest,
            },
            f,
            indent=2,
        )

    print(f"wrote: {out}")
    print(f"rows: {len(all_rows)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()