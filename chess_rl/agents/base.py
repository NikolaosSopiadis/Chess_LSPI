# chess_rl/agents/base.py
from __future__ import annotations
from dataclasses import dataclass
from chess_core.board import Board
from chess_core.move import Move

@dataclass
class AgentInfo:
    name: str
    version: str

class Agent:
    info: AgentInfo

    def pick_move(self, board: Board) -> Move:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Agent":
        raise NotImplementedError
