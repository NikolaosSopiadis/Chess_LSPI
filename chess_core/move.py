from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, IntFlag

class Promotion(IntEnum):
    NONE   = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK   = 3
    QUEEN  = 4

class MoveFlag(IntFlag):
    QUIET       = 0
    CAPTURE     = 1 << 0
    EN_PASSANT  = 1 << 1
    CASTLE      = 1 << 2
    PROMOTION   = 1 << 3
    DOUBLE_PAWN = 1 << 4

@dataclass(frozen=True)
class Move:
    src_square: int # idx of source square
    dst_square: int # idx of destination square

    flags: MoveFlag      = MoveFlag.QUIET
    promotion: Promotion = Promotion.NONE
    captured: int = 0 # idx of captrued piece

    @staticmethod
    def normal(src: int, dst: int, captured: int = 0) -> Move:
        return Move(src, dst,
                    MoveFlag.CAPTURE if captured else MoveFlag.QUIET,
                    Promotion.NONE, captured)

    @staticmethod
    def promotion_to(src: int, dst: int, to: Promotion, captured: int = 0) -> Move:
        flags = MoveFlag.PROMOTION
        if captured: flags |= MoveFlag.CAPTURE
        return Move(src, dst, flags, to, captured)

    @staticmethod
    def castle(src: int, dst: int) -> Move:
        return Move(src, dst, MoveFlag.CASTLE)

    @staticmethod
    def en_passant(src: int, dst: int) -> Move:
        return Move(src, dst, MoveFlag.EN_PASSANT | MoveFlag.CAPTURE)

    @staticmethod
    def double_pawn(src: int, dst: int) -> Move:
        return Move(src, dst, MoveFlag.DOUBLE_PAWN)