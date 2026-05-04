from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from chess_core.board import Board
from chess_core.piece import Piece as p
from chess_core.move import Move

from chess_rl.features.base import FeatureSpec


FloatArray = npt.NDArray[np.float64]


PIECE_VALUES_CP = {
    p.PAWN: 100,
    p.KNIGHT: 320,
    p.BISHOP: 330,
    p.ROOK: 500,
    p.QUEEN: 900,
    p.KING: 0,
}


class V2BasicFeatures:
    """
    v2_basic: white-perspective afterstate features.

    Convention:
      - Positive feature values generally mean "good for White".
      - White policy maximizes w·phi.
      - Black policy minimizes w·phi.

    Main purpose over v1_basic:
      - Make quiet positions less indistinguishable.
      - Add mobility/activity.
      - Add attacked/hanging material.
      - Add king pressure.
      - Add pawn progress.
      - Add halfmove-clock pressure.
    """

    spec = FeatureSpec(name="features", version="v2_basic", dim=32)

    def phi_sa(self, board: Board, move: Move) -> FloatArray:
        undo = board._do_move(move)
        try:
            return self.phi_afterstate(board)
        finally:
            board._undo_move(undo)

    def phi_afterstate(self, board: Board) -> FloatArray:
        x = np.zeros(self.spec.dim, dtype=np.float64)

        mat = _material_counts(board)
        (
            wp, wn, wb, wr, wq,
            bp, bn, bb, br, bq,
        ) = mat

        white_material_cp = (
            100 * wp +
            320 * wn +
            330 * wb +
            500 * wr +
            900 * wq
        )
        black_material_cp = (
            100 * bp +
            320 * bn +
            330 * bb +
            500 * br +
            900 * bq
        )

        material_diff_cp = white_material_cp - black_material_cp

        cr = board._castling_rights

        white_in_check = board.in_check(True)
        black_in_check = board.in_check(False)

        white_mob = _pseudo_mobility(board, True)
        black_mob = _pseudo_mobility(board, False)

        attacked = _attacked_material_features(board)
        pawns = _pawn_progress_features(board)
        king = _king_pressure_features(board)

        halfmove_clock = float(getattr(board, "_halfmove_clock", 0))

        # --- v1-ish base material/state features ---
        x[0] = 1.0
        x[1] = material_diff_cp / 1000.0
        x[2] = (wp - bp) / 8.0
        x[3] = (wn - bn) / 2.0
        x[4] = (wb - bb) / 2.0
        x[5] = (wr - br) / 2.0
        x[6] = float(wq - bq)

        x[7] = 1.0 if (cr & Board.WHITE_CASTLE_KINGSIDE) else 0.0
        x[8] = 1.0 if (cr & Board.WHITE_CASTLE_QUEENSIDE) else 0.0
        x[9] = 1.0 if (cr & Board.BLACK_CASTLE_KINGSIDE) else 0.0
        x[10] = 1.0 if (cr & Board.BLACK_CASTLE_QUEENSIDE) else 0.0

        # Centered side-to-move feature.
        x[11] = 1.0 if board.get_is_white_to_move() else -1.0

        # Both kings, not only the side-to-move king.
        x[12] = 1.0 if white_in_check else 0.0
        x[13] = 1.0 if black_in_check else 0.0

        # --- mobility / activity ---
        x[14] = (white_mob - black_mob) / 40.0
        x[15] = white_mob / 40.0
        x[16] = black_mob / 40.0

        # --- attacked and hanging material ---
        # Positive diff means good for White.
        x[17] = attacked["attacked_diff_cp"] / 1000.0
        x[18] = attacked["white_attacks_black_cp"] / 1000.0
        x[19] = attacked["black_attacks_white_cp"] / 1000.0

        x[20] = attacked["hanging_diff_cp"] / 1000.0
        x[21] = attacked["white_hanging_cp"] / 1000.0
        x[22] = attacked["black_hanging_cp"] / 1000.0

        # --- king pressure ---
        # Positive means White attacks Black king zone more than vice versa.
        x[23] = king["king_pressure_diff"] / 8.0
        x[24] = king["white_king_danger"] / 8.0
        x[25] = king["black_king_danger"] / 8.0

        # --- pawn progress ---
        x[26] = pawns["pawn_advancement_diff"] / 48.0
        x[27] = pawns["white_pawn_advancement"] / 48.0
        x[28] = pawns["black_pawn_advancement"] / 48.0
        x[29] = pawns["passed_pawn_diff"] / 8.0
        x[30] = pawns["promotion_pressure_diff"] / 8.0

        # Draw/progress pressure. High means closer to fifty-move draw.
        # The learner can use this together with material/features to prefer pawn moves/captures.
        x[31] = halfmove_clock / 100.0

        return x


def _material_counts(board: Board) -> list[int]:
    """
    Use Board's maintained material counts when available.

    Order:
      wp, wn, wb, wr, wq, bp, bn, bb, br, bq
    """
    mat = getattr(board, "_mat", None)
    if mat is not None:
        return list(mat)

    # Fallback, slower.
    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for pc in board.get_board():
        pc = int(pc)
        if pc == p.NONE:
            continue

        t = p.piece_type(pc)
        white = p.is_white(pc)

        if t == p.PAWN:
            out[0 if white else 5] += 1
        elif t == p.KNIGHT:
            out[1 if white else 6] += 1
        elif t == p.BISHOP:
            out[2 if white else 7] += 1
        elif t == p.ROOK:
            out[3 if white else 8] += 1
        elif t == p.QUEEN:
            out[4 if white else 9] += 1

    return out


def _piece_value_cp(pc: int) -> int:
    if pc == p.NONE:
        return 0
    return PIECE_VALUES_CP.get(p.piece_type(pc), 0)


def _pseudo_mobility(board: Board, white: bool) -> int:
    """
    Count pseudolegal moves for one side.

    This is deliberately cheaper than legal mobility. It is only an
    activity proxy, not a legality proof.
    """
    old_turn = board._is_white_to_move
    old_ep = board._en_passant_target

    try:
        board._is_white_to_move = white

        if white != old_turn:
            board._en_passant_target = None

        total = 0
        for src, pc in enumerate(board.get_board()):
            if pc == p.NONE:
                continue
            if p.is_white(pc) != white:
                continue
            total += len(board.get_pseudolegal_moves(src))

        return total

    finally:
        board._is_white_to_move = old_turn
        board._en_passant_target = old_ep


def _attacked_material_features(board: Board) -> dict[str, float]:
    """
    Compute simple attacked/hanging material features.

    attacked:
      enemy piece is attacked by this side.

    hanging:
      own non-king piece is attacked by enemy and not defended by own side.
    """
    white_attacks_black_cp = 0
    black_attacks_white_cp = 0

    white_hanging_cp = 0
    black_hanging_cp = 0

    bseq = board.get_board()

    for sq, pc0 in enumerate(bseq):
        pc = int(pc0)
        if pc == p.NONE:
            continue

        t = p.piece_type(pc)
        if t == p.KING:
            continue

        val = _piece_value_cp(pc)
        white_piece = p.is_white(pc)

        attacked_by_white = board.is_square_attacked(sq, by_white=True)
        attacked_by_black = board.is_square_attacked(sq, by_white=False)

        defended_by_white = attacked_by_white
        defended_by_black = attacked_by_black

        if white_piece:
            if attacked_by_black:
                black_attacks_white_cp += val

                if not defended_by_white:
                    white_hanging_cp += val
        else:
            if attacked_by_white:
                white_attacks_black_cp += val

                if not defended_by_black:
                    black_hanging_cp += val

    attacked_diff_cp = white_attacks_black_cp - black_attacks_white_cp

    # Positive means good for White:
    # black hanging material minus white hanging material.
    hanging_diff_cp = black_hanging_cp - white_hanging_cp

    return {
        "white_attacks_black_cp": float(white_attacks_black_cp),
        "black_attacks_white_cp": float(black_attacks_white_cp),
        "attacked_diff_cp": float(attacked_diff_cp),
        "white_hanging_cp": float(white_hanging_cp),
        "black_hanging_cp": float(black_hanging_cp),
        "hanging_diff_cp": float(hanging_diff_cp),
    }


def _king_zone_squares(king_sq: int) -> list[int]:
    if king_sq < 0:
        return []

    f0 = king_sq & 7
    r0 = king_sq >> 3

    out: list[int] = []
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            f = f0 + df
            r = r0 + dr
            if 0 <= f < 8 and 0 <= r < 8:
                out.append(f + (r << 3))

    return out


def _king_pressure_features(board: Board) -> dict[str, float]:
    white_king_sq = board._white_king_sq
    black_king_sq = board._black_king_sq

    white_king_danger = 0
    for sq in _king_zone_squares(white_king_sq):
        if board.is_square_attacked(sq, by_white=False):
            white_king_danger += 1

    black_king_danger = 0
    for sq in _king_zone_squares(black_king_sq):
        if board.is_square_attacked(sq, by_white=True):
            black_king_danger += 1

    return {
        "white_king_danger": float(white_king_danger),
        "black_king_danger": float(black_king_danger),
        "king_pressure_diff": float(black_king_danger - white_king_danger),
    }


def _pawn_progress_features(board: Board) -> dict[str, float]:
    bseq: Sequence[int] = board.get_board()

    white_pawn_advancement = 0
    black_pawn_advancement = 0

    white_passed = 0
    black_passed = 0

    white_promotion_pressure = 0
    black_promotion_pressure = 0

    white_pawns: list[int] = []
    black_pawns: list[int] = []

    for sq, pc0 in enumerate(bseq):
        pc = int(pc0)
        if pc == p.WHITE_PAWN:
            white_pawns.append(sq)
        elif pc == p.BLACK_PAWN:
            black_pawns.append(sq)

    for sq in white_pawns:
        r = sq >> 3
        # White starts at rank index 1. Larger rank means more advanced.
        white_pawn_advancement += r

        if r >= 5:
            white_promotion_pressure += 1

        if _is_passed_pawn(sq, True, black_pawns):
            white_passed += 1

    for sq in black_pawns:
        r = sq >> 3
        # Black starts at rank index 6. Smaller rank means more advanced.
        black_pawn_advancement += 7 - r

        if r <= 2:
            black_promotion_pressure += 1

        if _is_passed_pawn(sq, False, white_pawns):
            black_passed += 1

    return {
        "white_pawn_advancement": float(white_pawn_advancement),
        "black_pawn_advancement": float(black_pawn_advancement),
        "pawn_advancement_diff": float(white_pawn_advancement - black_pawn_advancement),
        "passed_pawn_diff": float(white_passed - black_passed),
        "promotion_pressure_diff": float(white_promotion_pressure - black_promotion_pressure),
    }


def _is_passed_pawn(sq: int, white: bool, enemy_pawns: list[int]) -> bool:
    f = sq & 7
    r = sq >> 3

    files = {f}
    if f > 0:
        files.add(f - 1)
    if f < 7:
        files.add(f + 1)

    for esq in enemy_pawns:
        ef = esq & 7
        er = esq >> 3

        if ef not in files:
            continue

        if white:
            # Enemy pawn ahead of white pawn.
            if er > r:
                return False
        else:
            # Enemy pawn ahead of black pawn.
            if er < r:
                return False

    return True