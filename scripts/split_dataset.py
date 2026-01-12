from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import TextIO

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


def split_jsonl_gz(src: str, out_dir: str, shards: int) -> list[str]:
    src_p = Path(src)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    outs: list[TextIO] = []
    out_paths: list[Path] = []
    for i in range(shards):
        p = out_p / f"{src_p.stem}.shard{i:03d}.jsonl.gz"
        out_paths.append(p)
        outs.append(gzip.open(p, "wt", encoding="utf-8"))

    try:
        with gzip.open(src_p, "rt", encoding="utf-8") as f:
            it = f
            if tqdm is not None:
                it = tqdm(f, desc="split", unit="lines", mininterval=0.25)
            for k, line in enumerate(it):
                line = line.strip()
                if not line:
                    continue
                outs[k % shards].write(line + "\n")
    finally:
        for o in outs:
            o.close()

    return [str(p) for p in out_paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Input .jsonl.gz dataset")
    ap.add_argument("--out-dir", required=True, help="Directory to write shard files")
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()

    paths = split_jsonl_gz(args.src, args.out_dir, args.shards)
    print("wrote shards:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
