from __future__ import annotations

from dataclasses import dataclass, field
import random

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent, AgentInfo


# Board._mat order:
# wp, wn, wb, wr, wq, bp, bn, bb, br, bq
_MATERIAL_VALUES = (
    100, 320, 330, 500, 900,
    -100, -320, -330, -500, -900,
)


def _material_score(board: Board) -> int:
    """
    Positive = better for White.
    Negative = better for Black.
    """
    return sum(c * v for c, v in zip(board._mat, _MATERIAL_VALUES))


@dataclass
class MaterialGreedyAgent(Agent):
    """
    Simple baseline:
    - White chooses the legal move with the highest immediate material score.
    - Black chooses the legal move with the lowest immediate material score.
    - Ties are random.
    """
    info: AgentInfo = field(default_factory=lambda: AgentInfo(name="MaterialGreedy", version="v1"))
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def pick_move(self, board: Board) -> Move:
        moves = board.get_all_legal_moves()
        if not moves:
            raise ValueError("No legal moves")

        white_to_move = board.get_is_white_to_move()

        best_score: int | None = None
        best_moves: list[Move] = []

        for move in moves:
            undo = board._do_move(move)
            try:
                score = _material_score(board)
            finally:
                board._undo_move(undo)

            if best_score is None:
                best_score = score
                best_moves = [move]
                continue

            if white_to_move:
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)
            else:
                if score < best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        return self._rng.choice(best_moves)

    def save(self, path: str) -> None:
        raise NotImplementedError("MaterialGreedyAgent has no checkpoint")

    @classmethod
    def load(cls, path: str) -> "MaterialGreedyAgent":
        return cls()