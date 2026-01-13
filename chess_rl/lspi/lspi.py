# chess_rl/lspi/lspi.py
from __future__ import annotations
import multiprocessing as mp
from multiprocessing.process import BaseProcess

from dataclasses import dataclass
import gzip
import json
import os
from typing import Iterator, Optional, cast, Any
from collections.abc import Callable
from collections import OrderedDict

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

from chess_core.board import Board
from chess_rl.features.base import FeatureExtractor
from chess_rl.policy.greedy import LegalMoveCache, greedy_choice

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore
    
try:
    import orjson  # type: ignore
except Exception:
    orjson = None  # type: ignore


CheckpointCB = Callable[[int, Float64Array, float], None]

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

@dataclass
class _ActionPhiCacheEntry:
    white_to_move: bool
    phis: Float32Array   # (M, d) float32 to save RAM


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
    move_cache: Optional[LegalMoveCache] = None
) -> tuple[Float64Array, Float64Array]:
    d = feats.spec.dim
    A: Float64Array = np.zeros((d, d), dtype=np.float64)
    b: Float64Array = np.zeros((d,), dtype=np.float64)

    b_next = Board()  # reuse to avoid reallocation

    base_iter: Iterator[dict] = iter_samples_jsonl_gz(samples_path)
    
    CHUNK = 4096
    Phi_buf = np.empty((CHUNK, d), dtype=np.float64)
    Phi_next_buf = np.empty((CHUNK, d), dtype=np.float64)
    r_buf = np.empty((CHUNK,), dtype=np.float64)
    diff_tmp = np.empty((CHUNK, d), dtype=np.float64)

    filled = 0
    n = 0

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

    for rec in it:
        if rec.get("feature_version") != feats.spec.version:
            raise ValueError(
                f"Feature version mismatch: dataset={rec.get('feature_version')!r} "
                f"vs feats={feats.spec.version!r}"
            )

        if rec.get("reward_version") not in (None, "v1_terminal_plus_potential"):
            raise ValueError("reward version mismatch ...")

        # Fill buffers
        Phi_buf[filled] = rec["phi"]
        r_buf[filled] = float(rec["r"])
        done = bool(rec["done"])

        if done:
            Phi_next_buf[filled].fill(0.0)
        else:
            b_next.init_board(str(rec["fen_next"]))
            Phi_next_buf[filled] = greedy_choice(b_next, w, feats, move_cache).phi

        filled += 1
        n += 1

        if filled == CHUNK:
            np.multiply(Phi_next_buf, cfg.gamma, out=diff_tmp)
            np.subtract(Phi_buf, diff_tmp, out=diff_tmp)
            A += Phi_buf.T @ diff_tmp
            b += Phi_buf.T @ r_buf
            filled = 0

        if cfg.max_samples is not None and n >= cfg.max_samples:
            break

    # flush tail
    if filled:
        Phi = Phi_buf[:filled]
        Phi_next = Phi_next_buf[:filled]
        r_vec = r_buf[:filled]
        dt = diff_tmp[:filled]
        np.multiply(Phi_next, cfg.gamma, out=dt)
        np.subtract(Phi, dt, out=dt)
        A += Phi.T @ dt
        b += Phi.T @ r_vec

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
    move_cache: Optional[LegalMoveCache] = None
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
            phi_next = greedy_choice(b_next, w, feats, move_cache).phi  # <- reuse best phi

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
    shard_paths: Optional[list[str]] = None,
    workers: Optional[int] = None, 
    feature_name: str = "v1_basic",
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
    if preload and not shard_paths:
        samples_mem = load_samples_mem(samples_path, feats, cfg, show_progress=verbose)


    I: Float64Array = cast(Float64Array, np.eye(d, dtype=np.float64))

    # --- create the pool once ---
    pool: Optional[PinnedShardPool] = None
    n_workers = workers or max(1, (os.cpu_count() or 1) - 1)
    
    # create move cache
    move_cache: LegalMoveCache = LegalMoveCache(max_size=250_000)

    try:
        if shard_paths:
            pool = PinnedShardPool(
                shard_paths,
                feature_name,
                cfg,
                workers=n_workers,
                preload=preload,
                action_cache=True,
                cache_max=200_000,
                move_cache_max=250_000,
            )

        for it in range(cfg.max_iters):
            if pool is not None:
                A, bvec = pool.accumulate(
                    w,
                    it + 1,
                    desc=f"LSPI pool accumulate {it+1}/{cfg.max_iters}",
                    verbose=verbose,
                )
            elif samples_mem is not None:
                A, bvec = _accumulate_A_b_mem(
                    samples_mem, w, feats, cfg,
                    show_progress=verbose,
                    desc=f"LSPI accumulate mem {it+1}/{cfg.max_iters}",
                    move_cache=move_cache,
                )
            else:
                A, bvec = _accumulate_A_b(
                    samples_path, w, feats, cfg,
                    show_progress=verbose,
                    total_hint=total_hint,
                    desc=f"LSPI accumulate {it+1}/{cfg.max_iters}",
                    move_cache=move_cache,
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
    finally:
        if pool is not None:
            pool.close()
            
    # print(f"Move cache: hits={move_cache.hits}, misses={move_cache.misses}, hit rate={move_cache.hits / max(1, move_cache.hits + move_cache.misses):.3%}")

    return w


def _load_shards_mem(
    shard_paths: list[str],
    feats: FeatureExtractor,
    cfg: LSPIConfig,
) -> SamplesMem:
    phis: list[Float64Array] = []
    rs: list[float] = []
    dones: list[bool] = []
    fens: list[str] = []

    n = 0
    for sp in shard_paths:
        for rec in iter_samples_jsonl_gz(sp):
            if rec.get("feature_version") != feats.spec.version:
                raise ValueError("feature version mismatch")
            if rec.get("reward_version") not in (None, "v1_terminal_plus_potential"):
                raise ValueError("reward version mismatch")

            phis.append(np.asarray(rec["phi"], dtype=np.float64))
            rs.append(float(rec["r"]))
            dones.append(bool(rec["done"]))
            fens.append(str(rec["fen_next"]))

            n += 1
            if cfg.max_samples is not None and n >= cfg.max_samples:
                break
        if cfg.max_samples is not None and n >= cfg.max_samples:
            break

    phi_mat = np.vstack(phis) if phis else np.zeros((0, feats.spec.dim), dtype=np.float64)
    r_arr = np.asarray(rs, dtype=np.float64)
    done_arr = np.asarray(dones, dtype=np.bool_)
    return SamplesMem(phi=phi_mat, r=r_arr, done=done_arr, fen_next=fens)


def _worker_loop_pinned(
    shard_paths: list[str],
    feature_name: str,
    cfg: LSPIConfig,
    in_q: "mp.Queue[Any]",
    out_q: "mp.Queue[Any]",
    *,
    preload: bool,
    action_cache: bool,
    cache_max: int,
    move_cache_max: int = 250_000,
) -> None:
    # imports inside subprocess
    from chess_rl.features.registry import get as get_features

    feats = get_features(feature_name)
    b_tmp = Board()
    move_cache = LegalMoveCache(max_size=move_cache_max)

    CHUNK = 4096  # tune: 2048/4096/8192
    Phi_next_buf = np.empty((CHUNK, feats.spec.dim), dtype=np.float64)
    diff_tmp = np.empty((CHUNK, feats.spec.dim), dtype=np.float64)

    # preload this worker's shard(s) once (big win: no gz/json every LSPI iter)
    samples_mem: Optional[SamplesMem] = None
    if preload:
        samples_mem = _load_shards_mem(shard_paths, feats, cfg)

    # optional LRU action-phi cache per worker
    # fen -> _ActionPhiCacheEntry, LRU via OrderedDict
    lru: "OrderedDict[str, _ActionPhiCacheEntry]" = OrderedDict()

    def best_phi_next_cached_local(fen: str, w32: Float32Array) -> Float32Array:
        k = fen_key(fen)
        entry = lru.get(k)
        if entry is None:
            b_tmp.init_board(fen)
            moves = b_tmp.get_all_legal_moves()
            if not moves:
                phis32 = np.zeros((1, feats.spec.dim), dtype=np.float32)
                entry = _ActionPhiCacheEntry(b_tmp.get_is_white_to_move(), phis32)
            else:
                phis32 = np.empty((len(moves), feats.spec.dim), dtype=np.float32)
                for i, m in enumerate(moves):
                    phis32[i] = feats.phi_sa(b_tmp, m)   # copies into row
                entry = _ActionPhiCacheEntry(b_tmp.get_is_white_to_move(), phis32)

            lru[k] = entry
            if cache_max > 0 and len(lru) > cache_max:
                lru.popitem(last=False)  # evict oldest
        else:
            # mark as most recently used
            lru.move_to_end(k)

        scores = entry.phis @ w32
        idx = int(np.argmax(scores) if entry.white_to_move else np.argmin(scores))
        # return np.asarray(entry.phis[idx], dtype=np.float64)
        return entry.phis[idx]

    while True:
        msg = in_q.get()
        if msg is None:
            return

        iter_idx, w = msg
        w32 = w.astype(np.float32)   # cast from float64 to float32
        d = feats.spec.dim
        A = np.zeros((d, d), dtype=np.float64)
        b = np.zeros((d,), dtype=np.float64)

        if samples_mem is not None:
            # fast path: mem
            N = samples_mem.phi.shape[0]
            for start in range(0, N, CHUNK):
                end = min(N, start + CHUNK)
                B = end - start

                Phi = samples_mem.phi[start:end]          # (B,d) view
                r_vec = samples_mem.r[start:end]          # (B,) view
                done_vec = samples_mem.done[start:end]    # (B,) view

                Phi_next = Phi_next_buf[:B]
                Phi_next.fill(0.0)

                if action_cache:
                    for j in range(B):
                        if done_vec[j]:
                            continue

                        fen = samples_mem.fen_next[start + j]
                        # write float32 -> float64 row (no allocation)
                        Phi_next[j] = best_phi_next_cached_local(fen, w32)
                else:
                    for j in range(B):
                        if done_vec[j]:
                            continue
                        fen = samples_mem.fen_next[start + j]
                        b_tmp.init_board(fen)
                        Phi_next[j] = greedy_choice(b_tmp, w, feats, move_cache).phi

                # BLAS update
                # diff_tmp[:B] = Phi - gamma*Phi_next
                np.multiply(Phi_next, cfg.gamma, out=diff_tmp[:B])
                np.subtract(Phi, diff_tmp[:B], out=diff_tmp[:B])

                A += Phi.T @ diff_tmp[:B]
                b += Phi.T @ r_vec

        else:
            # streaming path: gz/json each iter (chunked)
            Phi_buf = np.empty((CHUNK, d), dtype=np.float64)
            Phi_next_buf = np.empty((CHUNK, d), dtype=np.float64)
            r_buf = np.empty((CHUNK,), dtype=np.float64)
            filled = 0

            def flush(B: int) -> None:
                # diff_tmp[:B] = Phi_buf[:B] - gamma*Phi_next_buf[:B]
                _update_A_b_chunk(
                    A, b,
                    Phi_buf[:B],
                    Phi_next_buf[:B],
                    r_buf[:B],
                    cfg.gamma,
                    diff_tmp[:B],
                )

            for sp in shard_paths:
                for rec in iter_samples_jsonl_gz(sp):
                    Phi_buf[filled] = rec["phi"]
                    r_buf[filled] = float(rec["r"])
                    done = bool(rec["done"])

                    if done:
                        Phi_next_buf[filled].fill(0.0)
                    else:
                        fen = str(rec["fen_next"])
                        if action_cache:
                            Phi_next_buf[filled] = best_phi_next_cached_local(fen, w32)
                        else:
                            b_tmp.init_board(fen)
                            Phi_next_buf[filled] = greedy_choice(b_tmp, w, feats).phi

                    filled += 1
                    if filled == CHUNK:
                        flush(CHUNK)
                        filled = 0

            if filled:
                flush(filled)


        out_q.put((iter_idx, A, b))

# do not include fullmove number in fen key for better cache hits
def fen_key(fen: str) -> str:
    parts = fen.split()
    return " ".join(parts[:5])  # placement, active, castling, ep, halfmove


class PinnedShardPool:
    def __init__(
        self,
        shard_paths:    list[str],
        feature_name:   str,
        cfg:            LSPIConfig,
        *,
        workers:        int,
        preload:        bool = True,
        action_cache:   bool = False,
        cache_max:      int = 50_000,
        move_cache_max: int = 250_000,
    ) -> None:
        self.feature_name = feature_name
        self.cfg = cfg
        self.workers = workers
        self.preload = preload
        self.action_cache = action_cache
        self.cache_max = cache_max
        self.move_cache_max = move_cache_max

        shard_paths = sorted(shard_paths)
        self.buckets: list[list[str]] = [shard_paths[i::workers] for i in range(workers)]

        self.ctx = mp.get_context("fork")  # Linux: fastest
        self.in_queues: list[mp.Queue[Any]] = [self.ctx.Queue(maxsize=1) for _ in range(workers)]
        self.out_q: mp.Queue[Any] = self.ctx.Queue()

        self.procs: list[BaseProcess] = []
        for i in range(workers):
            p = self.ctx.Process(
                target=_worker_loop_pinned,
                args=(self.buckets[i], feature_name, cfg, self.in_queues[i], self.out_q),
                kwargs=dict(
                    preload=preload,
                    action_cache=action_cache,
                    cache_max=cache_max,
                    move_cache_max=move_cache_max
                ),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def close(self) -> None:
        for q in self.in_queues:
            q.put(None)
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()

    def accumulate(self, w: Float64Array,iter_idx: int, *, 
                   desc: str, verbose: bool) -> tuple[Float64Array, Float64Array]:
        # send work
        for q in self.in_queues:
            q.put((iter_idx, w))

        d = w.shape[0]
        A_total = np.zeros((d, d), dtype=np.float64)
        b_total = np.zeros((d,), dtype=np.float64)

        pbar = tqdm(total=self.workers, desc=desc, unit="workers", mininterval=0.25) if (verbose and tqdm is not None) else None

        got = 0
        while got < self.workers:
            i_idx, A_part, b_part = self.out_q.get()
            if i_idx != iter_idx:
                continue
            A_total += A_part
            b_total += b_part
            got += 1
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return A_total, b_total

def _update_A_b_chunk(
    A: Float64Array,
    b: Float64Array,
    Phi: Float64Array,          # (B,d)
    Phi_next: Float64Array,     # (B,d)
    r: Float64Array,            # (B,)
    gamma: float,
    diff_tmp: Float64Array,     # (B,d)
) -> None:
    # Diff = Phi - gamma*Phi_next
    np.multiply(Phi_next, gamma, out=diff_tmp)
    np.subtract(Phi, diff_tmp, out=diff_tmp)
    A += Phi.T @ diff_tmp
    b += Phi.T @ r
