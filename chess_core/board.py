import numpy as np
import numpy.typing as npt

from chess_core.piece import Piece as p

class Board:
    
    def __init__(self, height=8, width=8):
        self._height:     int = height
        self._width:      int = width
        self._grid_size: int = height * width

        self._board: npt.NDArray[np.uint8] = np.zeros(self._grid_size, dtype=np.uint8)
        self._board[0] = p.WHITE_ROOK
        self._board[1] = p.WHITE_KNIGHT
        self._board[2] = p.WHITE_BISHOP
        self._board[3] = p.WHITE_QUEEN
        self._board[4] = p.WHITE_KING
        