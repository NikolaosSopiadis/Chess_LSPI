from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from typing import Iterator, Optional

import numpy as np

from chess_core.board import Board
from chess_rl.features.base import FeatureExtractor
from chess_rl.policy.greedy import greedy_move


@dataclass(frozen=True)
class LSPIConfig:
    gamma: float = 0.99
    reg: float = 1e-3          # ridge term (lambda)
    max_iters: int = 20
    tol: float = 1e-6          # stop when ||w_new - w|| < tol
    max_samples: Optional[int] = None  # useful for quick dev runs


def iter_samples_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _accumulate_A_b(
    samples_path: str,
    w: np.ndarray,
    feats: FeatureExtractor,
    cfg: LSPIConfig,
) -> tuple[np.ndarray, np.ndarray]:
    d = feats.spec.dim
    A = np.zeros((d, d), dtype=np.float64)
    b = np.zeros((d,), dtype=np.float64)

    b_next = Board()  # reuse to avoid reallocation

    n = 0
    for rec in iter_samples_jsonl_gz(samples_path):
        if rec.get("feature_version") != feats.spec.version:
            raise ValueError(
                f"Feature version mismatch: dataset={rec.get('feature_version')!r} "
                f"vs feats={feats.spec.version!r}"
            )
        
        if rec.get("reward_version") not in (None, "v1_terminal_plus_potential"):
            raise ValueError("reward version mismatch ...")

        phi = np.asarray(rec["phi"], dtype=np.float64)
        r = float(rec["r"])
        done = bool(rec["done"])

        if done:
            phi_next = np.zeros(d, dtype=np.float64)
        else:
            fen_next = rec["fen_next"]
            b_next.init_board(fen_next)

            # policy improvement step: greedy under current weights
            a_next = greedy_move(b_next, w, feats)
            phi_next = feats.phi_sa(b_next, a_next)

        diff = phi - cfg.gamma * phi_next
        A += np.outer(phi, diff)
        b += phi * r

        n += 1
        if cfg.max_samples is not None and n >= cfg.max_samples:
            break

    return A, b


def solve_lspi(
    samples_path: str,
    feats: FeatureExtractor,
    cfg: LSPIConfig = LSPIConfig(),
    *,
    w0: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    d = feats.spec.dim
    w = np.zeros(d, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).copy()
    if w.shape != (d,):
        raise ValueError(f"w0 must have shape ({d},), got {w.shape}")

    I = np.eye(d, dtype=np.float64)

    for it in range(cfg.max_iters):
        A, bvec = _accumulate_A_b(samples_path, w, feats, cfg)

        # Regularize for stability
        A_reg = A + cfg.reg * I

        # Solve (A + λI) w = b
        try:
            w_new = np.linalg.solve(A_reg, bvec)
        except np.linalg.LinAlgError:
            # fallback: least squares
            w_new, *_ = np.linalg.lstsq(A_reg, bvec, rcond=None)

        delta = float(np.linalg.norm(w_new - w))
        w = w_new

        if verbose:
            print(f"[LSPI] iter {it+1}/{cfg.max_iters}  |Δw|={delta:.3e}")

        if delta < cfg.tol:
            if verbose:
                print("[LSPI] converged")
            break

    return w
