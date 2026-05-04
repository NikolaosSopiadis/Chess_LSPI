from __future__ import annotations

import numpy as np
import numpy.typing as npt

from chess_core.board import Board
from chess_core.move import Move

from chess_rl.features.base import FeatureSpec
from chess_rl.rewards.v1_terminal_plus_potential import material_potential

from chess_rl.features.v2_basic import (
    _material_counts,
    _pseudo_mobility,
    _attacked_material_features,
    _pawn_progress_features,
    _king_pressure_features,
)


FloatArray = npt.NDArray[np.float64]


_DRAW_REASONS = {
    "stalemate",
    "threefold repetition",
    "insufficient material",
    "fifty-move rule",
}


class V3BasicFeatures:
    """
    v3_basic: v2_1_basic + draw/conversion/repetition features.

    White-perspective convention:
      - White maximizes w·phi
      - Black minimizes w·phi
    """

    spec = FeatureSpec(name="features", version="v3_basic", dim=47)

    def phi_sa(self, board: Board, move: Move) -> FloatArray:
        with board.temporary_move(move):
            return self.phi_afterstate(board)

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

        white_pseudo_mob = _pseudo_mobility(board, True)
        black_pseudo_mob = _pseudo_mobility(board, False)

        white_legal_mob = board.legal_mobility(True)
        black_legal_mob = board.legal_mobility(False)

        side_legal_mob = (
            white_legal_mob
            if board.get_is_white_to_move()
            else black_legal_mob
        )

        attacked = _attacked_material_features(board)
        pawns = _pawn_progress_features(board)
        king = _king_pressure_features(board)

        halfmove_clock = float(getattr(board, "_halfmove_clock", 0))
        halfmove_pressure = min(1.0, halfmove_clock / 100.0)

        material = float(material_potential(board))
        white_ahead = max(material, 0.0)
        black_ahead = max(-material, 0.0)

        done, reason = board.game_end_state()

        terminal_draw = bool(done and reason in _DRAW_REASONS)
        terminal_checkmate = bool(done and reason == "checkmate")

        # If side to move is checkmated:
        #   white to move + checkmate => black wins
        #   black to move + checkmate => white wins
        white_wins = terminal_checkmate and not board.get_is_white_to_move()
        black_wins = terminal_checkmate and board.get_is_white_to_move()

        rep_count = board.current_repetition_count()
        repeat_risk = 1.0 if rep_count >= 2 else 0.0

        # Low enemy mobility while ahead can indicate conversion pressure,
        # but it can also indicate stalemate danger.
        white_enemy_low_mobility = max(0.0, 4.0 - black_legal_mob) / 4.0
        black_enemy_low_mobility = max(0.0, 4.0 - white_legal_mob) / 4.0

        # --- v2_1 base features ---
        x[0] = 1.0

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

        x[13] = (white_pseudo_mob - black_pseudo_mob) / 40.0
        x[14] = white_pseudo_mob / 40.0
        x[15] = black_pseudo_mob / 40.0

        x[16] = attacked["attacked_diff_cp"] / 1000.0
        x[17] = attacked["white_attacks_black_cp"] / 1000.0
        x[18] = attacked["black_attacks_white_cp"] / 1000.0

        x[19] = attacked["hanging_diff_cp"] / 1000.0
        x[20] = attacked["white_hanging_cp"] / 1000.0
        x[21] = attacked["black_hanging_cp"] / 1000.0

        x[22] = king["king_pressure_diff"] / 8.0
        x[23] = king["white_king_danger"] / 8.0
        x[24] = king["black_king_danger"] / 8.0

        x[25] = pawns["pawn_advancement_diff"] / 48.0
        x[26] = pawns["white_pawn_advancement"] / 48.0
        x[27] = pawns["black_pawn_advancement"] / 48.0
        x[28] = pawns["passed_pawn_diff"] / 8.0
        x[29] = pawns["promotion_pressure_diff"] / 8.0

        x[30] = halfmove_clock / 100.0

        # --- v3 draw/conversion features ---
        x[31] = white_legal_mob / 40.0
        x[32] = black_legal_mob / 40.0
        x[33] = (white_legal_mob - black_legal_mob) / 40.0
        x[34] = side_legal_mob / 40.0

        x[35] = 1.0 if terminal_draw else 0.0
        x[36] = 1.0 if white_wins else 0.0
        x[37] = 1.0 if black_wins else 0.0

        # Drawing while ahead should be bad. Drawing while behind can be good.
        x[38] = white_ahead if terminal_draw else 0.0
        x[39] = black_ahead if terminal_draw else 0.0

        # Repetition history features.
        x[40] = min(float(rep_count), 3.0) / 3.0
        x[41] = white_ahead if repeat_risk else 0.0
        x[42] = black_ahead if repeat_risk else 0.0

        # Fifty-move pressure while ahead.
        x[43] = white_ahead * halfmove_pressure
        x[44] = black_ahead * halfmove_pressure

        # Enemy has few legal moves while we are ahead.
        # This may help learn stalemate/conversion patterns.
        x[45] = white_ahead * white_enemy_low_mobility
        x[46] = black_ahead * black_enemy_low_mobility

        return x