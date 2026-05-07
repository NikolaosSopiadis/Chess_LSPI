from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from chess_core.board import Board
from chess_core.move import F_CAPTURE, Move
from chess_core.piece import Piece as p

from chess_rl.features.v6_attackmap import V6AttackMapFeatures


@dataclass(frozen=True)
class _FeatureSpec:
    version: str
    dim: int


@dataclass(slots=True)
class _SideTactics:
    checking_moves: float = 0.0
    safe_checking_moves: float = 0.0
    mate_in_one_moves: float = 0.0
    safe_capture_value: float = 0.0
    unsafe_capture_liability: float = 0.0


class V7ApiTacticsFeatures:
    """
    v7_api_tactics = v6_attackmap + legal-move tactical/API features.

    Sign convention:
      positive = good for White
      negative = good for Black

    Base:
      0..56 = v6_attackmap

    Added:
      57 legal_checking_moves_diff
      58 safe_checking_moves_diff
      59 mate_in_one_threat_diff
      60 safe_capture_value_diff
      61 unsafe_capture_liability_diff
      62 queen_tempo_threat_diff
      63 queen_tempo_threat_by_minor_or_pawn_diff
    """

    spec = _FeatureSpec(version="v7_api_tactics", dim=64)

    PAWN_OR_MINOR_TYPES = frozenset({p.PAWN, p.KNIGHT, p.BISHOP})

    PIECE_VALUE = {
        p.PAWN: 1.0,
        p.KNIGHT: 3.0,
        p.BISHOP: 3.0,
        p.ROOK: 5.0,
        p.QUEEN: 9.0,
        p.KING: 0.0,
    }

    def __init__(self) -> None:
        self._base = V6AttackMapFeatures()

    def phi_afterstate(self, board: Board) -> np.ndarray:
        base = np.asarray(self._base.phi_afterstate(board), dtype=np.float64)

        if base.shape[0] != 57:
            raise ValueError(
                f"V7ApiTacticsFeatures expected v6_attackmap dim 57, got {base.shape[0]}"
            )

        extra = self._api_tactics_features(board)

        out = np.concatenate([base, extra]).astype(np.float64, copy=False)

        if out.shape[0] != self.spec.dim:
            raise ValueError(f"v7_api_tactics produced dim {out.shape[0]}, expected {self.spec.dim}")

        return out

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        with board.temporary_move(move):
            return self.phi_afterstate(board)

    # ------------------------------------------------------------------
    # Main extra feature block
    # ------------------------------------------------------------------

    def _api_tactics_features(self, board: Board) -> np.ndarray:
        white = self._side_tactics(board, white=True)
        black = self._side_tactics(board, white=False)

        queen_tempo = self._queen_tempo_threat_diff(
            board,
            piece_types=None,
            divisor=8.0,
        )

        queen_tempo_pawn_minor = self._queen_tempo_threat_diff(
            board,
            piece_types=self.PAWN_OR_MINOR_TYPES,
            divisor=6.0,
        )

        return np.array(
            [
                self._diff_norm(white.checking_moves, black.checking_moves, 16.0),
                self._diff_norm(white.safe_checking_moves, black.safe_checking_moves, 10.0),
                self._diff_norm(white.mate_in_one_moves, black.mate_in_one_moves, 2.0),
                self._diff_norm(white.safe_capture_value, black.safe_capture_value, 20.0),

                # Positive means Black has more unsafe capture liability.
                self._diff_norm(black.unsafe_capture_liability, white.unsafe_capture_liability, 20.0),

                queen_tempo,
                queen_tempo_pawn_minor,
            ],
            dtype=np.float64,
        )

    def _side_tactics(self, board: Board, *, white: bool) -> _SideTactics:
        out = _SideTactics()

        with board.temporary_side_to_move(white):
            moves = board.get_all_legal_moves()

            for move in moves:
                moving_piece = board.piece_at(move.src_square)
                moving_type = p.piece_type(moving_piece)
                moving_value = self.PIECE_VALUE.get(moving_type, 0.0)

                gives_check = board.move_gives_check(move)

                if gives_check:
                    out.checking_moves += 1.0

                    if self._landing_square_safe_after(board, move, mover_white=white):
                        out.safe_checking_moves += 1.0

                    if board.move_gives_checkmate(move):
                        out.mate_in_one_moves += 1.0

                if move.flags & F_CAPTURE:
                    captured_piece = self._captured_piece_before_move(board, move)
                    captured_type = p.piece_type(captured_piece)
                    captured_value = self.PIECE_VALUE.get(captured_type, 0.0)

                    if captured_value <= 0.0:
                        continue

                    if self._landing_square_safe_after(board, move, mover_white=white):
                        out.safe_capture_value += captured_value
                    else:
                        # A crude SEE-like penalty:
                        # capturing a pawn with a queen and losing the queen is very bad,
                        # capturing a queen with a knight even if recaptured can still be fine.
                        liability = max(0.0, moving_value - captured_value)
                        out.unsafe_capture_liability += liability

        return out

    # ------------------------------------------------------------------
    # Queen tempo threats
    # ------------------------------------------------------------------

    def _queen_tempo_threat_diff(
        self,
        board: Board,
        *,
        piece_types: frozenset[int] | None,
        divisor: float,
    ) -> float:
        """
        Positive means White has more legal moves that attack Black's queen.
        Negative means Black has more legal moves that attack White's queen.

        This is the feature meant to punish early exposed queen play.
        """
        white_queen = board.queen_square(True)
        black_queen = board.queen_square(False)

        white_moves_attacking_black_queen = 0
        black_moves_attacking_white_queen = 0

        if black_queen is not None:
            white_moves_attacking_black_queen = len(
                board.legal_moves_attacking_square(
                    white=True,
                    square=black_queen,
                    piece_types=piece_types,
                )
            )

        if white_queen is not None:
            black_moves_attacking_white_queen = len(
                board.legal_moves_attacking_square(
                    white=False,
                    square=white_queen,
                    piece_types=piece_types,
                )
            )

        return self._diff_norm(
            white_moves_attacking_black_queen,
            black_moves_attacking_white_queen,
            divisor,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _landing_square_safe_after(
        self,
        board: Board,
        move: Move,
        *,
        mover_white: bool,
    ) -> bool:
        """
        True if the moved piece's destination is not attacked by the opponent
        after the move.

        This is intentionally simple and not a full SEE.
        """
        with board.temporary_move(move):
            return not board.is_square_attacked(move.dst_square, by_white=not mover_white)

    def _captured_piece_before_move(self, board: Board, move: Move) -> int:
        """
        Return captured piece. Handles normal captures and en-passant if Move
        stores captured_piece.
        """
        captured_piece = int(getattr(move, "captured_piece", p.NONE))

        if captured_piece != p.NONE:
            return captured_piece

        return board.piece_at(move.dst_square)

    def _diff_norm(self, white_value: float, black_value: float, divisor: float) -> float:
        if divisor <= 0.0:
            raise ValueError("divisor must be positive")

        return self._clamp(float(white_value - black_value) / divisor, -1.0, 1.0)

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))