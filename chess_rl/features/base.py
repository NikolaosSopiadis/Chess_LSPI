from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from chess_core.board import Board
from chess_core.move import Move


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    version: str
    dim: int


class FeatureExtractor(Protocol):
    spec: FeatureSpec

    def phi_sa(self, board: Board, move: Move) -> FloatArray:
        ...

    def phi_afterstate(self, board_after: Board) -> FloatArray:
        ...