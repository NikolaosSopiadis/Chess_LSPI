# chess_rl/lspi/lspi.py
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from typing import Iterator, Optional, cast
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

from chess_core.board import Board
from chess_rl.features.base import FeatureExtractor
from chess_rl.policy.greedy import greedy_choice

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore
    
try:
    import orjson  # type: ignore
except Exception:
    orjson = None  # type: ignore


CheckpointCB = Callable[[int, npt.NDArray[np.float64], float], None]

@dataclass(frozen=True)
class LSPIConfig:
    gamma: float = 0.99
    reg: float = 1e-3
    max_iters: int = 20
    tol: float = 1e-6
    max_samples: Optional[int] = None  # dev runs
    
@dataclass(frozen=True, slots=True)
class SamplesMem:
    phi: Float64Array          # (N, d)
    r: Float64Array            # (N,)
    done: npt.NDArray[np.bool_]
    fen_next: list[str]        # length N



def load_samples_mem(
    samples_path: str,
    feats: FeatureExtractor,
    cfg: LSPIConfig,
    *,
    show_progress: bool = True,
) -> SamplesMem:
    phis: list[Float64Array] = []
    rs: list[float] = []
    dones: list[bool] = []
    fens: list[str] = []

    it = iter_samples_jsonl_gz(samples_path)
    pbar = tqdm(it, unit="rows", desc="preload samples", mininterval=0.25) if (show_progress and tqdm is not None) else None
    it2 = pbar if pbar is not None else it

    n = 0
    for rec in it2:
        if rec.get("feature_version") != feats.spec.version:
            raise ValueError(
                f"Feature version mismatch: dataset={rec.get('feature_version')!r} "
                f"vs feats={feats.spec.version!r}"
            )
        if rec.get("reward_version") not in (None, "v1_terminal_plus_potential"):
            raise ValueError("reward version mismatch ...")

        phis.append(np.asarray(rec["phi"], dtype=np.float64))
        rs.append(float(rec["r"]))
        dones.append(bool(rec["done"]))
        fens.append(str(rec["fen_next"]))

        n += 1
        if cfg.max_samples is not None and n >= cfg.max_samples:
            break

    if pbar is not None:
        pbar.close()

    phi_mat = np.vstack(phis) if phis else np.zeros((0, feats.spec.dim), dtype=np.float64)
    r_arr = np.asarray(rs, dtype=np.float64)
    done_arr = np.asarray(dones, dtype=np.bool_)

    return SamplesMem(phi=phi_mat, r=r_arr, done=done_arr, fen_next=fens)

def iter_samples_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if orjson is not None:
                yield orjson.loads(line)
            else:
                yield json.loads(line)


def _count_rows_jsonl_gz(path: str) -> int:
    """
    Count rows once to enable ETA when max_samples is None.
    This costs one extra full pass through the gz file.
    """
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _accumulate_A_b(
    samples_path: str,
    w: Float64Array,
    feats: FeatureExtractor,
    cfg: "LSPIConfig",
    *,
    show_progress: bool,
    total_hint: Optional[int],
    desc: str,
) -> tuple[Float64Array, Float64Array]:
    d = feats.spec.dim
    A: Float64Array = np.zeros((d, d), dtype=np.float64)
    b: Float64Array = np.zeros((d,), dtype=np.float64)

    b_next = Board()  # reuse to avoid reallocation

    base_iter: Iterator[dict] = iter_samples_jsonl_gz(samples_path)

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(
            base_iter,
            total=total_hint,
            unit="rows",
            desc=desc,
            leave=False,
            mininterval=0.25,
        )
        it = pbar
    else:
        it = base_iter

    n = 0
    for rec in it:
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

            choice = greedy_choice(b_next, w, feats)
            phi_next = choice.phi

        diff = phi - cfg.gamma * phi_next
        # A += np.outer(phi, diff)
        # Replace the above line with this to avoid a (d,d) temporary:
        # rank-1 update without creating a (d,d) temporary each row
        for i in range(d):
            A[i] += phi[i] * diff
        b += phi * r

        n += 1
        if cfg.max_samples is not None and n >= cfg.max_samples:
            break

    if pbar is not None:
        pbar.close()

    return A, b

def _accumulate_A_b_mem(
    samples: SamplesMem,
    w: Float64Array,
    feats: FeatureExtractor,
    cfg: LSPIConfig,
    *,
    show_progress: bool,
    desc: str,
) -> tuple[Float64Array, Float64Array]:
    d = feats.spec.dim
    A: Float64Array = np.zeros((d, d), dtype=np.float64)
    b: Float64Array = np.zeros((d,), dtype=np.float64)

    b_next = Board()

    idx_iter = range(samples.phi.shape[0])
    pbar = tqdm(idx_iter, total=samples.phi.shape[0], unit="rows", desc=desc, leave=False, mininterval=0.25) if (show_progress and tqdm is not None) else None
    it = pbar if pbar is not None else idx_iter

    for i in it:
        phi = samples.phi[i]
        r = float(samples.r[i])
        done = bool(samples.done[i])

        if done:
            phi_next = np.zeros(d, dtype=np.float64)
        else:
            b_next.init_board(samples.fen_next[i])
            phi_next = greedy_choice(b_next, w, feats).phi  # <- reuse best phi

        diff = phi - cfg.gamma * phi_next

        # keep your rank-1 update (or revert to outer if you prefer)
        for k in range(d):
            A[k] += phi[k] * diff

        b += phi * r

    if pbar is not None:
        pbar.close()

    return A, b

def solve_lspi(
    samples_path: str,
    feats: FeatureExtractor,
    cfg: LSPIConfig = LSPIConfig(),
    *,
    w0: Optional[np.ndarray] = None,
    verbose: bool = True,
    checkpoint_cb: Optional[CheckpointCB] = None,
    preload: bool = False,
) -> np.ndarray:
    d = feats.spec.dim
    w_arr = np.zeros(d, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).copy()
    w: Float64Array = cast(Float64Array, w_arr)
    if w.shape != (d,):
        raise ValueError(f"w0 must have shape ({d},), got {w.shape}")

    # For ETA:
    # - if max_samples is set: tqdm total is known => ETA works immediately
    # - else: we optionally count file rows once (enables ETA for full runs too)
    total_hint: Optional[int] = cfg.max_samples
    if total_hint is None and verbose and tqdm is not None:
        # One-time scan to enable ETA on full dataset runs
        total_hint = _count_rows_jsonl_gz(samples_path)
        
    samples_mem: Optional[SamplesMem] = None
    if preload:
        samples_mem = load_samples_mem(samples_path, feats, cfg, show_progress=verbose)

    I: Float64Array = cast(Float64Array, np.eye(d, dtype=np.float64))

    for it in range(cfg.max_iters):
        if samples_mem is not None:
            A, bvec = _accumulate_A_b_mem(
                samples_mem,
                w,
                feats,
                cfg,
                show_progress=verbose,
                desc=f"LSPI accumulate mem {it+1}/{cfg.max_iters}",
            )
        else:
            A, bvec = _accumulate_A_b(
                samples_path,
                w,
                feats,
                cfg,
                show_progress=verbose,
                total_hint=total_hint,
                desc=f"LSPI accumulate {it+1}/{cfg.max_iters}",
            )

        A_reg = A + cfg.reg * I

        try:
            w_new = np.linalg.solve(A_reg, bvec)
        except np.linalg.LinAlgError:
            w_new, *_ = np.linalg.lstsq(A_reg, bvec, rcond=None)

        w_new = cast(Float64Array, np.asarray(w_new, dtype=np.float64))
        delta = float(np.linalg.norm(w_new - w))
        w = w_new

        if verbose:
            if tqdm is not None:
                tqdm.write(f"[LSPI] iter {it+1}/{cfg.max_iters}  |Δw|={delta:.3e}")
            else:
                print(f"[LSPI] iter {it+1}/{cfg.max_iters}  |Δw|={delta:.3e}")
                
        if checkpoint_cb is not None:
            checkpoint_cb(it + 1, w, delta)

        if delta < cfg.tol:
            if verbose:
                if tqdm is not None:
                    tqdm.write("[LSPI] converged")
                else:
                    print("[LSPI] converged")
            break

    return w
