from __future__ import annotations

from dataclasses import dataclass, field
import random

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent, AgentInfo


@dataclass
class RandomAgent(Agent):
    info: AgentInfo = field(default_factory=lambda: AgentInfo(name="Random", version="v1"))
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def pick_move(self, board: Board) -> Move:
        moves = board.get_all_legal_moves()
        if not moves:
            raise ValueError("No legal moves")
        return self._rng.choice(moves)

    def save(self, path: str) -> None:
        raise NotImplementedError("RandomAgent has no checkpoint")

    @classmethod
    def load(cls, path: str) -> "RandomAgent":
        return cls()