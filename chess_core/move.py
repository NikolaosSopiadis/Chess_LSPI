from __future__ import annotations
from dataclasses import dataclass

# Changed from Enum to IntFlag because of huge enum overhead
# promotion codes (plain ints)
PROMO_NONE   = 0
PROMO_KNIGHT = 1
PROMO_BISHOP = 2
PROMO_ROOK   = 3
PROMO_QUEEN  = 4

# flag bits (plain ints)
F_QUIET       = 0
F_CAPTURE     = 1 << 0
F_EN_PASSANT  = 1 << 1
F_CASTLE      = 1 << 2
F_PROMOTION   = 1 << 3
F_DOUBLE_PAWN = 1 << 4

@dataclass(frozen=True, slots=True)
class Move:
    src_square: int # idx of source square
    dst_square: int # idx of destination square

    flags: int     = F_QUIET
    promotion: int = PROMO_NONE
    captured_piece: int = 0

    @staticmethod
    def normal(src: int, dst: int, captured: int = 0) -> Move:
        return Move(src, dst,
                    F_CAPTURE if captured else F_QUIET,
                    PROMO_NONE, captured)

    @staticmethod
    def promotion_to(src: int, dst: int, to: int, captured: int = 0) -> Move:
        flags = F_PROMOTION
        if captured: flags |= F_CAPTURE
        return Move(src, dst, flags, to, captured)

    @staticmethod
    def castle(src: int, dst: int) -> Move:
        return Move(src, dst, F_CASTLE)

    @staticmethod
    def en_passant(src: int, dst: int) -> Move:
        return Move(src, dst, F_EN_PASSANT | F_CAPTURE)
        
    @staticmethod
    def double_pawn(src: int, dst: int) -> Move:
        return Move(src, dst, F_DOUBLE_PAWN)

    def check_flag(self, flag: int) -> bool:
        return (self.flags & flag) == flag