from __future__ import annotations

from typing import Optional
import numpy as np
import numpy.typing as npt

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.features.base import FeatureExtractor


def greedy_move(board: Board, w: npt.NDArray[np.float64], feats: FeatureExtractor) -> Move:
    moves: list[Move] = board.get_all_legal_moves()
    if not moves:
        raise ValueError("No legal moves")
    
    if w.ndim != 1:
        raise ValueError("w must be 1D")

    white_to_move: bool = board.get_is_white_to_move()

    best: Optional[Move] = None
    best_score: float = 0.0

    for m in moves:
        phi = feats.phi_sa(board, m)  # expected shape (d,)
        score: float = float(w @ phi)

        if best is None:
            best = m
            best_score = score
            continue

        # weights represent value from White perspective:
        # white chooses max, black chooses min
        if white_to_move:
            if score > best_score:
                best = m
                best_score = score
        else:
            if score < best_score:
                best = m
                best_score = score

    assert best is not None
    return best
