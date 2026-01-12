# chess_rl/data/build_samples_puzzles.py
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
import os
from typing import Optional, Set, Iterable, Iterator, List, Dict, Any
from itertools import repeat

from tqdm import tqdm

from chess_core.board import Board
from chess_rl.uci import uci_to_move
from chess_rl.data.lichess_puzzles import iter_lichess_puzzles_csv, PuzzleRow
from chess_rl.features.registry import get as get_features
from chess_rl.rewards.v1_terminal_plus_potential import RewardSpec, material_potential


def _terminal_reward_from_after(after: Board, done: bool, reason: str) -> float:
    if not done:
        return 0.0
    if reason == "checkmate":
        # side to move is checkmated => loses
        loser_is_white = after.get_is_white_to_move()
        return -1.0 if loser_is_white else +1.0
    return 0.0


def _process_one_puzzle(
    row: PuzzleRow,
    feature_name: str,
    reward_alpha: float,
) -> List[Dict[str, Any]]:
    """
    Worker: convert one puzzle into 1+ LSPI samples (one per ply in PV).
    Returns a list of JSON-serializable dict records.
    """
    feats = get_features(feature_name)
    reward_version = RewardSpec().version

    b = Board()
    b.init_board(row.fen)

    out: List[Dict[str, Any]] = []

    for uci in row.moves_uci:
        done0, _ = b.game_end_state()
        if done0:
            break

        # reward potential BEFORE applying the move (avoid allocating a second Board)
        pot0 = material_potential(b)

        move = uci_to_move(b, uci)

        # features for (s,a): afterstate features (uses do/undo internally)
        phi = feats.phi_sa(b, move)

        # advance to next
        b._do_move(move)

        done, reason = b.game_end_state()
        pot1 = material_potential(b)

        r_terminal = _terminal_reward_from_after(b, done, reason)
        r = float(r_terminal + reward_alpha * (pot1 - pot0))

        rec = {
            "phi": phi.tolist(),
            "r": r,
            "fen_next": b.to_fen(),
            "done": bool(done),

            # versions / provenance
            "feature_version": feats.spec.version,
            "reward_version": reward_version,
            "source": "lichess_puzzles",
            "puzzle_id": row.puzzle_id,
            "themes": sorted(row.themes),
            "rating": int(row.rating),
        }
        out.append(rec)

        if done:
            break

    return out


def build_samples_from_puzzles(
    puzzles_csv: str,
    out_jsonl_gz: str,
    *,
    feature_name: str = "v1_basic",
    reward_alpha: float = 0.05,
    include_themes: Optional[Set[str]] = None,
    min_rating: int = 1800,
    max_rows: int = 50_000,
    workers: int = 0,
    chunksize: int = 32,
) -> None:
    """
    Build LSPI samples from the Lichess puzzles CSV.

    - Shows ETA with tqdm.
    - Parallelizes per-puzzle processing with ProcessPoolExecutor.
    - Writes a single gzip JSONL file from the main process.
    """
    rows_iter = iter_lichess_puzzles_csv(
        puzzles_csv,
        include_themes=include_themes,
        min_rating=min_rating,
        max_rows=max_rows,
    )

    total = max_rows if max_rows is not None else None
    if workers is None or workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    with gzip.open(out_jsonl_gz, "wt", encoding="utf-8") as f:
        # If workers==1, keep it simple (still with tqdm)
        if workers == 1:
            for row in tqdm(rows_iter, total=total, unit="puzzle"):
                recs = _process_one_puzzle(row, feature_name, reward_alpha)
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
            return

        # Parallel: each worker returns list[records] for a puzzle
        with ProcessPoolExecutor(max_workers=workers) as ex:
            mapped = ex.map(
                _process_one_puzzle,
                rows_iter,
                repeat(feature_name),
                repeat(reward_alpha),
                chunksize=chunksize,
            )

            for recs in tqdm(mapped, total=total, unit="puzzle"):
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
