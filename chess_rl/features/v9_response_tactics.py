from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from chess_core.board import Board
from chess_core.move import F_CAPTURE, F_CASTLE, Move
from chess_core.piece import Piece as p

from chess_rl.features.v8_api_tactics_clean import V8ApiTacticsCleanFeatures


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
    best_safe_capture_value: float = 0.0
    unsafe_capture_liability: float = 0.0

    queen_tempo_minor_or_pawn: float = 0.0


class V9ResponseTacticsFeatures:
    """
    v9_response_tactics = v8_api_tactics_clean + concrete tactical components
    + side-to-move response features + opening queen-exposure features.

    Sign convention:
      positive = good for White
      negative = good for Black
    """

    spec = _FeatureSpec(version="v9_response_tactics", dim=80)

    PAWN_OR_MINOR_TYPES = frozenset({p.PAWN, p.KNIGHT, p.BISHOP})

    PIECE_VALUE = {
        p.PAWN: 1.0,
        p.KNIGHT: 3.0,
        p.BISHOP: 3.0,
        p.ROOK: 5.0,
        p.QUEEN: 9.0,
        p.KING: 0.0,
    }

    WHITE_HOME_MINORS = (1, 2, 5, 6)       # b1 c1 f1 g1
    BLACK_HOME_MINORS = (57, 58, 61, 62)   # b8 c8 f8 g8

    def __init__(self) -> None:
        self._base = V8ApiTacticsCleanFeatures()

    def phi_afterstate(self, board: Board) -> np.ndarray:
        base = np.asarray(self._base.phi_afterstate(board), dtype=np.float64)

        if base.shape[0] != 44:
            raise ValueError(f"v9 expected v8 dim 44, got {base.shape[0]}")

        extra = self._extra_features(board)
        out = np.concatenate([base, extra]).astype(np.float64, copy=False)

        if out.shape[0] != self.spec.dim:
            raise ValueError(f"v9 produced dim {out.shape[0]}, expected {self.spec.dim}")

        return out

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        with board.temporary_move(move):
            return self.phi_afterstate(board)

    def _extra_features(self, board: Board) -> np.ndarray:
        white = self._side_tactics(board, white=True)
        black = self._side_tactics(board, white=False)

        stm_white = board.get_is_white_to_move()
        ntm_white = not stm_white

        stm = white if stm_white else black
        ntm = black if stm_white else white

        stm_sign = 1.0 if stm_white else -1.0
        ntm_sign = 1.0 if ntm_white else -1.0

        (
            white_q_exp,
            black_q_exp,
            white_dev_deficit,
            black_dev_deficit,
            white_king_center,
            black_king_center,
            white_castle_ready,
            black_castle_ready,
        ) = self._opening_features(board)

        return np.array(
            [
                # ----------------------------------------------------------
                # Concrete components behind v8 tactical diffs
                # ----------------------------------------------------------
                self._norm(white.checking_moves, 16.0),
                self._norm(black.checking_moves, 16.0),

                self._norm(white.safe_checking_moves, 10.0),
                self._norm(black.safe_checking_moves, 10.0),

                self._norm(white.mate_in_one_moves, 2.0),
                self._norm(black.mate_in_one_moves, 2.0),

                self._norm(white.safe_capture_value, 20.0),
                self._norm(black.safe_capture_value, 20.0),

                self._norm(white.best_safe_capture_value, 9.0),
                self._norm(black.best_safe_capture_value, 9.0),

                self._norm(white.unsafe_capture_liability, 20.0),
                self._norm(black.unsafe_capture_liability, 20.0),

                self._norm(white.queen_tempo_minor_or_pawn, 6.0),
                self._norm(black.queen_tempo_minor_or_pawn, 6.0),

                # ----------------------------------------------------------
                # Side-to-move immediate response features
                # Positive means good for White, negative good for Black.
                # In an afterstate, STM is the player who replies next.
                # ----------------------------------------------------------
                stm_sign * self._norm(stm.safe_capture_value, 20.0),
                stm_sign * self._norm(stm.best_safe_capture_value, 9.0),
                stm_sign * self._norm(stm.safe_checking_moves, 10.0),
                stm_sign * self._norm(stm.mate_in_one_moves, 2.0),
                stm_sign * self._norm(stm.queen_tempo_minor_or_pawn, 6.0),

                # ----------------------------------------------------------
                # Not-side-to-move latent threat features
                # This is the player who just made the candidate move.
                # ----------------------------------------------------------
                ntm_sign * self._norm(ntm.safe_capture_value, 20.0),
                ntm_sign * self._norm(ntm.best_safe_capture_value, 9.0),
                ntm_sign * self._norm(ntm.safe_checking_moves, 10.0),
                ntm_sign * self._norm(ntm.mate_in_one_moves, 2.0),
                ntm_sign * self._norm(ntm.queen_tempo_minor_or_pawn, 6.0),

                # ----------------------------------------------------------
                # Opening / queen-exposure concrete features
                # ----------------------------------------------------------
                white_q_exp,
                black_q_exp,
                black_q_exp - white_q_exp,

                white_dev_deficit,
                black_dev_deficit,
                black_dev_deficit - white_dev_deficit,

                white_king_center,
                black_king_center,
                black_king_center - white_king_center,

                white_castle_ready,
                black_castle_ready,
                white_castle_ready - black_castle_ready,
            ],
            dtype=np.float64,
        )

    def _side_tactics(self, board: Board, *, white: bool) -> _SideTactics:
        out = _SideTactics()

        with board.temporary_side_to_move(white):
            moves = board.get_all_legal_moves()

            enemy_queen = board.queen_square(not white)

            for move in moves:
                moving_piece = board.piece_at(move.src_square)
                moving_type = p.piece_type(moving_piece)
                moving_value = self.PIECE_VALUE.get(moving_type, 0.0)

                if board.move_gives_check(move):
                    out.checking_moves += 1.0

                    if self._landing_square_safe_after(board, move, mover_white=white):
                        out.safe_checking_moves += 1.0

                    if board.move_gives_checkmate(move):
                        out.mate_in_one_moves += 1.0

                if move.flags & F_CAPTURE:
                    captured_piece = self._captured_piece_before_move(board, move)
                    captured_type = p.piece_type(captured_piece)
                    captured_value = self.PIECE_VALUE.get(captured_type, 0.0)

                    if captured_value > 0.0:
                        if self._landing_square_safe_after(board, move, mover_white=white):
                            out.safe_capture_value += captured_value
                            out.best_safe_capture_value = max(
                                out.best_safe_capture_value,
                                captured_value,
                            )
                        else:
                            out.unsafe_capture_liability += max(
                                0.0,
                                moving_value - captured_value,
                            )

                if enemy_queen is not None:
                    if moving_type in self.PAWN_OR_MINOR_TYPES:
                        if self._move_attacks_square_after(
                            board,
                            move,
                            square=enemy_queen,
                            by_white=white,
                        ):
                            out.queen_tempo_minor_or_pawn += 1.0

        return out

    def _opening_features(self, board: Board) -> tuple[float, float, float, float, float, float, float, float]:
        phase = self._opening_phase(board)

        white_dev_deficit = phase * self._unfinished_development(board, True)
        black_dev_deficit = phase * self._unfinished_development(board, False)

        white_king_center = phase * self._king_still_center(board, True)
        black_king_center = phase * self._king_still_center(board, False)

        white_castle_ready = phase * self._castle_ready(board, True)
        black_castle_ready = phase * self._castle_ready(board, False)

        white_q_exp = phase * self._queen_exposure(
            board,
            white=True,
            dev_deficit=white_dev_deficit,
        )
        black_q_exp = phase * self._queen_exposure(
            board,
            white=False,
            dev_deficit=black_dev_deficit,
        )

        return (
            white_q_exp,
            black_q_exp,
            white_dev_deficit,
            black_dev_deficit,
            white_king_center,
            black_king_center,
            white_castle_ready,
            black_castle_ready,
        )

    def _opening_phase(self, board: Board) -> float:
        """
        Crude phase gate.

        Avoids punishing queen activity in queen endgames.
        """
        nonpawn_nonking = 0

        for pc_raw in board.get_board():
            pc = int(pc_raw)
            if pc == p.NONE:
                continue

            t = p.piece_type(pc)
            if t not in (p.PAWN, p.KING):
                nonpawn_nonking += 1

        both_queens = board.queen_square(True) is not None and board.queen_square(False) is not None

        if not both_queens:
            return 0.0

        return self._clamp(nonpawn_nonking / 10.0, 0.0, 1.0)

    def _unfinished_development(self, board: Board, white: bool) -> float:
        home = self.WHITE_HOME_MINORS if white else self.BLACK_HOME_MINORS
        own_knight = p.WHITE_KNIGHT if white else p.BLACK_KNIGHT
        own_bishop = p.WHITE_BISHOP if white else p.BLACK_BISHOP

        count = 0
        for sq in home:
            pc = board.piece_at(sq)
            if pc == own_knight or pc == own_bishop:
                count += 1

        return count / 4.0

    def _king_still_center(self, board: Board, white: bool) -> float:
        k = board.king_square(white)

        if white:
            return 1.0 if k == 4 else 0.0   # e1

        return 1.0 if k == 60 else 0.0      # e8

    def _castle_ready(self, board: Board, white: bool) -> float:
        with board.temporary_side_to_move(white):
            for move in board.get_all_legal_moves():
                if move.flags & F_CASTLE:
                    return 1.0

        return 0.0

    def _queen_exposure(self, board: Board, *, white: bool, dev_deficit: float) -> float:
        qsq = board.queen_square(white)
        if qsq is None:
            return 0.0

        queen_home = 3 if white else 59  # d1/d8
        queen_out = 1.0 if qsq != queen_home else 0.0

        enemy_minor_pawn_attacks = len(
            board.legal_moves_attacking_square(
                white=not white,
                square=qsq,
                piece_types=self.PAWN_OR_MINOR_TYPES,
            )
        )

        attacked_component = self._norm(enemy_minor_pawn_attacks, 4.0)

        # Queen being out is bad mostly when development is unfinished.
        return self._clamp(
            0.45 * queen_out
            + 0.35 * queen_out * dev_deficit
            + 0.20 * attacked_component,
            0.0,
            1.0,
        )

    def _move_attacks_square_after(
        self,
        board: Board,
        move: Move,
        *,
        square: int,
        by_white: bool,
    ) -> bool:
        with board.temporary_move(move):
            return board.is_square_attacked(square, by_white=by_white)

    def _landing_square_safe_after(
        self,
        board: Board,
        move: Move,
        *,
        mover_white: bool,
    ) -> bool:
        with board.temporary_move(move):
            return not board.is_square_attacked(move.dst_square, by_white=not mover_white)

    def _captured_piece_before_move(self, board: Board, move: Move) -> int:
        captured_piece = int(getattr(move, "captured_piece", p.NONE))

        if captured_piece != p.NONE:
            return captured_piece

        return board.piece_at(move.dst_square)

    def _norm(self, value: float, divisor: float) -> float:
        return self._clamp(float(value) / divisor, 0.0, 1.0)

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))