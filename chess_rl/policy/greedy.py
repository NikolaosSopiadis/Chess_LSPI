# chess_rl/policy/greedy.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.features.base import FeatureExtractor

Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GreedyChoice:
    move: Move
    phi: Float64Array
    score: float

def greedy_choice(board: Board, w: Float64Array, feats: FeatureExtractor) -> GreedyChoice:
    z = board._zkey
    moves: Sequence[Move] 

    moves = board.get_all_legal_moves()

    if not moves:
        raise ValueError("No legal moves")
        
    if w.ndim != 1:
        raise ValueError("w must be 1D")

    white_to_move: bool = board.get_is_white_to_move()

    best_move: Optional[Move] = None
    best_phi: Optional[Float64Array] = None
    best_score: float = 0.0

    for m in moves:
        phi = feats.phi_sa(board, m)
        score = float(w @ phi)

        if best_move is None:
            best_move, best_phi, best_score = m, phi, score
            continue

        if white_to_move:
            if score > best_score:
                best_move, best_phi, best_score = m, phi, score
        else:
            if score < best_score:
                best_move, best_phi, best_score = m, phi, score

    assert best_move is not None and best_phi is not None
    return GreedyChoice(best_move, best_phi, best_score)


def greedy_move(board: Board, w: Float64Array, feats: FeatureExtractor) -> Move:
    return greedy_choice(board, w, feats).move
