from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Sample:
    phi: np.ndarray
    r: float
    fen_next: str
    done: bool
