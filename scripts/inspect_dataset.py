# scripts/inspect_dataset.py
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from chess_rl.features.registry import get as get_features


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


class RunningStats:
    """Welford running mean/std for scalars."""
    __slots__ = ("n", "mean", "M2", "minv", "maxv")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0
        self.minv: float = float("inf")
        self.maxv: float = float("-inf")

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if x < self.minv:
            self.minv = x
        if x > self.maxv:
            self.maxv = x

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.n - 1))


def _try_tqdm(total: Optional[int]):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(total=total, unit="rows")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Path to samples.jsonl or samples.jsonl.gz")
    ap.add_argument("--features", default="v1_basic", help="Feature registry key (default: v1_basic)")
    ap.add_argument("--max-rows", type=int, default=None, help="Stop after N rows (for quick checks)")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm even if installed")
    ap.add_argument("--topk", type=int, default=10, help="How many metadata entries to show per field")
    args = ap.parse_args()

    feats = get_features(args.features)
    d_expected = feats.spec.dim
    ver_expected = feats.spec.version

    n_rows = 0
    n_done = 0
    n_bad_phi_dim = 0
    n_nan_phi = 0
    n_nan_r = 0
    n_missing = 0

    r_stats = RunningStats()

    feature_versions = Counter()
    reward_versions = Counter()
    sources = Counter()

    # Optional progress bar (unknown total)
    pbar = None if args.no_progress else _try_tqdm(total=args.max_rows)

    for rec in iter_jsonl(args.samples):
        n_rows += 1
        if pbar:
            pbar.update(1)

        # Required keys (minimal)
        missing_keys = [k for k in ("phi", "r", "done") if k not in rec]
        if missing_keys:
            n_missing += 1
            # keep going; but record it
            if args.max_rows is None and n_missing <= 5:
                print(f"[warn] row {n_rows}: missing keys {missing_keys}")
            if args.max_rows is not None and n_rows >= args.max_rows:
                break
            continue

        # Versions / provenance (optional but expected in your pipeline)
        fv = rec.get("feature_version", None)
        rv = rec.get("reward_version", None)
        src = rec.get("source", None)
        if fv is not None:
            feature_versions[str(fv)] += 1
        if rv is not None:
            reward_versions[str(rv)] += 1
        if src is not None:
            sources[str(src)] += 1

        # Check done
        done = bool(rec["done"])
        if done:
            n_done += 1

        # Reward checks
        try:
            r = float(rec["r"])
            if not math.isfinite(r):
                n_nan_r += 1
            else:
                r_stats.add(r)
        except Exception:
            n_nan_r += 1

        # Phi checks
        phi = rec["phi"]
        try:
            # quick dim check without heavy conversion
            if not isinstance(phi, list) or len(phi) != d_expected:
                n_bad_phi_dim += 1
            else:
                arr = np.asarray(phi, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    n_nan_phi += 1
        except Exception:
            n_nan_phi += 1

        # Feature version mismatch warning (don’t hard-fail in inspector)
        if fv is not None and str(fv) != ver_expected and n_rows <= 5:
            print(f"[warn] feature_version mismatch in early row: dataset={fv!r} expected={ver_expected!r}")

        if args.max_rows is not None and n_rows >= args.max_rows:
            break

    if pbar:
        pbar.close()

    print("\n=== Dataset inspection ===")
    print(f"file: {args.samples}")
    print(f"features: {args.features} (expected version={ver_expected!r}, dim={d_expected})")
    print(f"rows read: {n_rows:,}")
    print(f"done: {n_done:,} ({(n_done / n_rows * 100.0) if n_rows else 0.0:.2f}%)")
    print(f"missing required fields: {n_missing:,}")
    print(f"bad phi dim: {n_bad_phi_dim:,}")
    print(f"phi NaN/Inf or parse errors: {n_nan_phi:,}")
    print(f"reward NaN/Inf or parse errors: {n_nan_r:,}")

    if r_stats.n:
        print("\n--- Reward stats (finite only) ---")
        print(f"count: {r_stats.n:,}")
        print(f"mean:  {r_stats.mean:.6f}")
        print(f"std:   {r_stats.std():.6f}")
        print(f"min:   {r_stats.minv:.6f}")
        print(f"max:   {r_stats.maxv:.6f}")

    def show_counter(title: str, c: Counter) -> None:
        if not c:
            return
        print(f"\n--- {title} (top {args.topk}) ---")
        for k, v in c.most_common(args.topk):
            print(f"{k:>30} : {v:,}")

    show_counter("feature_version", feature_versions)
    show_counter("reward_version", reward_versions)
    show_counter("source", sources)

    # Return a non-zero exit code if there are serious issues (optional)
    # Here we just print. If you want strict mode later, we can add --strict.


if __name__ == "__main__":
    main()
