# chess_rl/features/v10_response_fast.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from chess_core.board import Board
from chess_core.move import F_CAPTURE, Move
from chess_core.piece import Piece as p

from chess_rl.features.v4_slim import V4SlimFeatures


@dataclass(frozen=True)
class _FeatureSpec:
    version: str
    dim: int


@dataclass(slots=True)
class _SideResponseStats:
    legal_checking_moves: float = 0.0
    safe_checking_moves: float = 0.0
    mate_in_one_moves: float = 0.0

    safe_capture_value: float = 0.0
    best_safe_capture_value: float = 0.0


class V10ResponseFastFeatures:
    """
    v10_response_fast = v4_slim + selected concrete tactical/response features.

    This is a speed-pruned v9:
      - removes v6 attack-map block
      - removes old v8/v9 diff tactical block
      - removes queen-tempo legal-move-attacking-square features
      - removes low-impact opening/static extras
      - keeps concrete checking/capture/side-to-move response features

    Sign convention:
      positive = good for White
      negative = good for Black

    Base:
      0..36 = v4_slim

    Added:
      37 white_legal_checking_moves
      38 black_legal_checking_moves
      39 white_safe_checking_moves
      40 black_safe_checking_moves
      41 white_mate_in_one_moves
      42 black_mate_in_one_moves
      43 white_safe_capture_value
      44 black_safe_capture_value
      45 white_best_safe_capture_value
      46 black_best_safe_capture_value
      47 stm_safe_capture_value_pm
      48 stm_best_safe_capture_value_pm
      49 stm_safe_checking_moves_pm
      50 stm_mate_in_one_moves_pm
      51 ntm_safe_capture_value_pm
      52 ntm_best_safe_capture_value_pm
      53 ntm_safe_checking_moves_pm
      54 ntm_mate_in_one_moves_pm
    """

    spec = _FeatureSpec(version="v10_response_fast", dim=55)

    PIECE_VALUE = {
        p.PAWN: 1.0,
        p.KNIGHT: 3.0,
        p.BISHOP: 3.0,
        p.ROOK: 5.0,
        p.QUEEN: 9.0,
        p.KING: 0.0,
    }

    def __init__(self) -> None:
        self._base = V4SlimFeatures()

    def phi_afterstate(self, board: Board) -> np.ndarray:
        base = np.asarray(self._base.phi_afterstate(board), dtype=np.float64)

        if base.shape[0] != 37:
            raise ValueError(
                f"V10ResponseFastFeatures expected v4_slim dim 37, got {base.shape[0]}"
            )

        extra = self._response_features(board)

        out = np.concatenate([base, extra]).astype(np.float64, copy=False)

        if out.shape[0] != self.spec.dim:
            raise ValueError(
                f"v10_response_fast produced dim {out.shape[0]}, expected {self.spec.dim}"
            )

        return out

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        with board.temporary_move(move):
            return self.phi_afterstate(board)

    # ------------------------------------------------------------------
    # Main feature block
    # ------------------------------------------------------------------

    def _response_features(self, board: Board) -> np.ndarray:
        white = self._side_response_stats(board, white=True)
        black = self._side_response_stats(board, white=False)

        stm_white = board.get_is_white_to_move()
        ntm_white = not stm_white

        stm = white if stm_white else black
        ntm = black if stm_white else white

        return np.array(
            [
                # Concrete side features.
                self._norm(white.legal_checking_moves, 16.0),
                self._norm(black.legal_checking_moves, 16.0),

                self._norm(white.safe_checking_moves, 10.0),
                self._norm(black.safe_checking_moves, 10.0),

                self._norm(white.mate_in_one_moves, 2.0),
                self._norm(black.mate_in_one_moves, 2.0),

                self._norm(white.safe_capture_value, 20.0),
                self._norm(black.safe_capture_value, 20.0),

                self._norm(white.best_safe_capture_value, 9.0),
                self._norm(black.best_safe_capture_value, 9.0),

                # Side-to-move perspective features.
                self._signed_for_color(
                    self._norm(stm.safe_capture_value, 20.0),
                    white=stm_white,
                ),
                self._signed_for_color(
                    self._norm(stm.best_safe_capture_value, 9.0),
                    white=stm_white,
                ),
                self._signed_for_color(
                    self._norm(stm.safe_checking_moves, 10.0),
                    white=stm_white,
                ),
                self._signed_for_color(
                    self._norm(stm.mate_in_one_moves, 2.0),
                    white=stm_white,
                ),

                # Non-side-to-move / opponent-response features.
                self._signed_for_color(
                    self._norm(ntm.safe_capture_value, 20.0),
                    white=ntm_white,
                ),
                self._signed_for_color(
                    self._norm(ntm.best_safe_capture_value, 9.0),
                    white=ntm_white,
                ),
                self._signed_for_color(
                    self._norm(ntm.safe_checking_moves, 10.0),
                    white=ntm_white,
                ),
                self._signed_for_color(
                    self._norm(ntm.mate_in_one_moves, 2.0),
                    white=ntm_white,
                ),
            ],
            dtype=np.float64,
        )

    def _side_response_stats(self, board: Board, *, white: bool) -> _SideResponseStats:
        out = _SideResponseStats()

        with board.temporary_side_to_move(white):
            moves = board.get_all_legal_moves()

            # Avoid duplicate temporary_move() calls for checking captures.
            safe_landing_cache: dict[tuple[int, int, int, int, int], bool] = {}

            for move in moves:
                moving_piece = board.piece_at(move.src_square)
                moving_type = p.piece_type(moving_piece)
                moving_value = self.PIECE_VALUE.get(moving_type, 0.0)

                gives_check = board.move_gives_check(move)

                if gives_check:
                    out.legal_checking_moves += 1.0

                    if self._landing_square_safe_after_cached(
                        board,
                        move,
                        mover_white=white,
                        cache=safe_landing_cache,
                    ):
                        out.safe_checking_moves += 1.0

                    if board.move_gives_checkmate(move):
                        out.mate_in_one_moves += 1.0

                if move.flags & F_CAPTURE:
                    captured_piece = self._captured_piece_before_move(board, move)

                    if captured_piece == p.NONE:
                        continue

                    captured_type = p.piece_type(captured_piece)
                    captured_value = self.PIECE_VALUE.get(captured_type, 0.0)

                    if captured_value <= 0.0:
                        continue

                    if self._landing_square_safe_after_cached(
                        board,
                        move,
                        mover_white=white,
                        cache=safe_landing_cache,
                    ):
                        out.safe_capture_value += captured_value
                        out.best_safe_capture_value = max(
                            out.best_safe_capture_value,
                            captured_value,
                        )
                    else:
                        # v10 intentionally does not expose unsafe-capture liability.
                        # It was weak in v9 and adds noise more than signal.
                        _ = moving_value

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _landing_square_safe_after_cached(
        self,
        board: Board,
        move: Move,
        *,
        mover_white: bool,
        cache: dict[tuple[int, int, int, int, int], bool],
    ) -> bool:
        key = (
            move.src_square,
            move.dst_square,
            int(move.flags),
            int(move.promotion),
            int(getattr(move, "captured_piece", p.NONE)),
        )

        cached = cache.get(key)
        if cached is not None:
            return cached

        safe = self._landing_square_safe_after(
            board,
            move,
            mover_white=mover_white,
        )
        cache[key] = safe
        return safe

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
        captured_piece = int(getattr(move, "captured_piece", p.NONE))

        if captured_piece != p.NONE:
            return captured_piece

        return board.piece_at(move.dst_square)

    def _norm(self, value: float, divisor: float) -> float:
        if divisor <= 0.0:
            raise ValueError("divisor must be positive")

        return self._clamp(float(value) / divisor, 0.0, 1.0)

    def _signed_for_color(self, value: float, *, white: bool) -> float:
        return float(value) if white else -float(value)

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))