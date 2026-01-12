# chess_rl/uci.py
from __future__ import annotations
from chess_core.board import Board
from chess_core.move import Move, Promotion

_UCI_PROMO = {
    "n": Promotion.KNIGHT,
    "b": Promotion.BISHOP,
    "r": Promotion.ROOK,
    "q": Promotion.QUEEN,
}

def uci_to_move(board: Board, uci: str) -> Move:
    """
    Convert a UCI move like 'e2e4' or 'e7e8q' into your exact Move
    (including flags such as CASTLE / EN_PASSANT / DOUBLE_PAWN).
    """
    uci = uci.strip()
    if len(uci) not in (4, 5):
        raise ValueError(f"Bad UCI: {uci!r}")

    src_sq = board.algebraic_to_idx(uci[0:2])
    dst_sq = board.algebraic_to_idx(uci[2:4])

    promo = Promotion.NONE
    if len(uci) == 5:
        ch = uci[4].lower()
        if ch not in _UCI_PROMO:
            raise ValueError(f"Bad UCI promotion: {uci!r}")
        promo = _UCI_PROMO[ch]

    # Let the engine decide the exact flags; we just select the matching legal move.
    for m in board.get_legal_moves(src_sq):
        if m.dst_square != dst_sq:
            continue
        if promo != Promotion.NONE and m.promotion != promo:
            continue
        if promo == Promotion.NONE and m.promotion != Promotion.NONE:
            continue
        return m

    raise ValueError(f"Illegal/unknown UCI in this position: {uci!r} (fen={board.to_fen()!r})")
