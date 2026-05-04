from __future__ import annotations

import numpy as np
import numpy.typing as npt

from chess_core.board import Board
from chess_core.move import Move

from chess_rl.features.base import FeatureSpec

from chess_rl.features.v2_basic import (
    _material_counts,
    _pseudo_mobility,
    _attacked_material_features,
    _pawn_progress_features,
    _king_pressure_features,
)


FloatArray = npt.NDArray[np.float64]


class V21BasicFeatures:
    """
    v2_1_basic: v2 without redundant aggregate material_total.

    Compared to v2_basic:
      - removes material_total
      - keeps individual piece count differences
      - keeps mobility, attacked material, hanging material, king pressure,
        pawn progress, and halfmove clock
    """

    spec = FeatureSpec(name="features", version="v2_1_basic", dim=31)

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

        cr = board._castling_rights

        white_in_check = board.in_check(True)
        black_in_check = board.in_check(False)

        white_mob = _pseudo_mobility(board, True)
        black_mob = _pseudo_mobility(board, False)

        attacked = _attacked_material_features(board)
        pawns = _pawn_progress_features(board)
        king = _king_pressure_features(board)

        halfmove_clock = float(getattr(board, "_halfmove_clock", 0))

        # --- material / state ---
        x[0] = 1.0

        # No aggregate material_total in v2_1.
        x[1] = (wp - bp) / 8.0
        x[2] = (wn - bn) / 2.0
        x[3] = (wb - bb) / 2.0
        x[4] = (wr - br) / 2.0
        x[5] = float(wq - bq)

        x[6] = 1.0 if (cr & Board.WHITE_CASTLE_KINGSIDE) else 0.0
        x[7] = 1.0 if (cr & Board.WHITE_CASTLE_QUEENSIDE) else 0.0
        x[8] = 1.0 if (cr & Board.BLACK_CASTLE_KINGSIDE) else 0.0
        x[9] = 1.0 if (cr & Board.BLACK_CASTLE_QUEENSIDE) else 0.0

        x[10] = 1.0 if board.get_is_white_to_move() else -1.0

        x[11] = 1.0 if white_in_check else 0.0
        x[12] = 1.0 if black_in_check else 0.0

        # --- mobility / activity ---
        x[13] = (white_mob - black_mob) / 40.0
        x[14] = white_mob / 40.0
        x[15] = black_mob / 40.0

        # --- attacked and hanging material ---
        x[16] = attacked["attacked_diff_cp"] / 1000.0
        x[17] = attacked["white_attacks_black_cp"] / 1000.0
        x[18] = attacked["black_attacks_white_cp"] / 1000.0

        x[19] = attacked["hanging_diff_cp"] / 1000.0
        x[20] = attacked["white_hanging_cp"] / 1000.0
        x[21] = attacked["black_hanging_cp"] / 1000.0

        # --- king pressure ---
        x[22] = king["king_pressure_diff"] / 8.0
        x[23] = king["white_king_danger"] / 8.0
        x[24] = king["black_king_danger"] / 8.0

        # --- pawn progress ---
        x[25] = pawns["pawn_advancement_diff"] / 48.0
        x[26] = pawns["white_pawn_advancement"] / 48.0
        x[27] = pawns["black_pawn_advancement"] / 48.0
        x[28] = pawns["passed_pawn_diff"] / 8.0
        x[29] = pawns["promotion_pressure_diff"] / 8.0

        # Draw/progress pressure.
        x[30] = halfmove_clock / 100.0

        return x