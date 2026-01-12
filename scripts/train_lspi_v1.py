from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from chess_rl.features.registry import get as get_features
from chess_rl.lspi.lspi import LSPIConfig, solve_lspi
from chess_rl.agents.lspi_v1 import LSPIV1Agent
from chess_rl.agents.base import AgentInfo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Path to samples.jsonl.gz")
    ap.add_argument("--out", required=True, help="Output .npz path (checkpoint)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    feats = get_features("v1_basic")
    cfg = LSPIConfig(
        gamma=args.gamma,
        reg=args.reg,
        max_iters=args.iters,
        tol=args.tol,
        max_samples=args.max_samples,
    )

    t0 = time.time()
    w = solve_lspi(args.samples, feats, cfg, verbose=True)
    dt = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    agent = LSPIV1Agent(
        info=AgentInfo(name="LSPI", version="v1"),
        w=w,
        feature_name="v1_basic",
    )
    agent.save(str(out))

    print(f"Saved checkpoint: {out}")
    print(f"Training time: {dt:.1f}s")
    print(f"w[:5] = {w[:5]}")


if __name__ == "__main__":
    main()
