# Move struct
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Move:
    # TODO: convert to flags
    src_square: int # idx of source square
    dst_square: int # idx of destination square

    en_passant_capture: bool = False
    castle: bool             = False
    pawn_two_up: bool        = False

    PROMOTE_TO_NONE: ClassVar[int]   = 0
    PROMOTE_TO_KNIGHT: ClassVar[int] = 1
    PROMOTE_TO_BISHOP: ClassVar[int] = 2
    PROMOTE_TO_ROOK: ClassVar[int]   = 3
    PROMOTE_TO_QUEEN: ClassVar[int]  = 4

    
    promotion_type: int = PROMOTE_TO_NONE
