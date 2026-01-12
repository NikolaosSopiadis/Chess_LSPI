from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from chess_core.board import Board
from chess_core.move import Move

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    dim: int

class FeatureExtractor(Protocol):
    spec: FeatureSpec
    def phi_sa(self, board: Board, move: Move) -> np.ndarray: ...
