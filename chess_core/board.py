import numpy as np
import numpy.typing as npt

from chess_core.piece import Piece as p
from chess_core.move import Move

class Board:
    
    CASTLE_LEFT: int  = 0
    CASTLE_RIGHT: int = 1
    CASTLE_WHITE: int = 0
    CASTLE_BLACK: int = 2
    
    WHITE_CASTLE_LEFT: int  = CASTLE_WHITE | CASTLE_LEFT
    WHITE_CASTLE_RIGHT: int = CASTLE_WHITE | CASTLE_RIGHT
    BLACK_CASTLE_LEFT: int  = CASTLE_BLACK | CASTLE_LEFT
    BLACK_CASTLE_RIGHT: int = CASTLE_BLACK | CASTLE_RIGHT
    
    DOUBLE_PAWN_MOVES_WHITE: int = 0
    DOUBLE_PAWN_MOVES_WHITE: int = 1
    
    def __init__(self, ranks=8, files=8):
        self._ranks:     int = ranks
        self._files:     int = files
        self._grid_size: int = ranks * files

        self._board: npt.NDArray[np.uint8] = np.zeros(self._grid_size, dtype=np.uint8)
        
        # 0x: white, 1x:black
        # x0: left, x1: right
        self._can__castle: list[bool] = [True, True, True, True] 
        # self._can_pawn_move_two_up[color][file]
        self._can_pawn_move_two_up: npt.NDArray[np.bool] = np.full((2,self._grid_size), True)
        
    def make_move(self, move: Move) -> bool:
        src: int = Move.src_square
        dst: int = Move.dst_square
        
        self._board[dst] = self._board[src]
        self._board[src] = p.NONE 
        
        return True