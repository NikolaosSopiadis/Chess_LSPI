from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from chess_core.board import Board
from chess_core.move import Move
from chess_core.piece import Piece as p

from chess_rl.features.v4_slim import V4SlimFeatures


@dataclass(frozen=True)
class _FeatureSpec:
    version: str
    dim: int


class V5CenterFeatures:
    """
    v5_center = v4_slim + explicit center/opening-center features.

    Sign convention:
      Positive feature values should generally be good for White.
      Negative feature values should generally be good for Black.

    Base:
      0..36  = v4_slim

    Added:
      37 center_pawn_presence_diff
      38 extended_center_pawn_presence_diff
      39 center_piece_occupation_diff
      40 center_control_diff
      41 extended_center_control_diff
      42 center_minor_control_diff
      43 center_pawn_control_diff
      44 opening_center_control_diff
      45 opening_center_pawn_presence_diff
      46 queen_out_before_center_diff
      47 white_queen_out_before_center
      48 black_queen_out_before_center
    """

    spec = _FeatureSpec(version="v5_center", dim=49)

    CORE_CENTER = ("d4", "e4", "d5", "e5")

    EXTENDED_CENTER = (
        "c3", "d3", "e3", "f3",
        "c4", "d4", "e4", "f4",
        "c5", "d5", "e5", "f5",
        "c6", "d6", "e6", "f6",
    )

    WHITE_CENTER_PAWN_WEIGHTS = {
        "d4": 1.0,
        "e4": 1.0,
        "c4": 0.5,
        "f4": 0.5,
        "d3": 0.5,
        "e3": 0.5,
    }

    BLACK_CENTER_PAWN_WEIGHTS = {
        "d5": 1.0,
        "e5": 1.0,
        "c5": 0.5,
        "f5": 0.5,
        "d6": 0.5,
        "e6": 0.5,
    }

    WHITE_EXTENDED_PAWN_SQUARES = (
        "c3", "d3", "e3", "f3",
        "c4", "d4", "e4", "f4",
    )

    BLACK_EXTENDED_PAWN_SQUARES = (
        "c6", "d6", "e6", "f6",
        "c5", "d5", "e5", "f5",
    )

    WHITE_MINOR_HOME = {
        "b1": p.KNIGHT,
        "g1": p.KNIGHT,
        "c1": p.BISHOP,
        "f1": p.BISHOP,
    }

    BLACK_MINOR_HOME = {
        "b8": p.KNIGHT,
        "g8": p.KNIGHT,
        "c8": p.BISHOP,
        "f8": p.BISHOP,
    }

    def __init__(self) -> None:
        self._base = V4SlimFeatures()

    def phi_afterstate(self, board: Board) -> np.ndarray:
        base = np.asarray(self._base.phi_afterstate(board), dtype=np.float64)

        if base.shape[0] != 37:
            raise ValueError(
                f"V5CenterFeatures expected v4_slim dim 37, got {base.shape[0]}"
            )

        extra = self._center_features(board)

        out = np.concatenate([base, extra]).astype(np.float64, copy=False)

        if out.shape[0] != self.spec.dim:
            raise ValueError(f"v5_center produced dim {out.shape[0]}, expected {self.spec.dim}")

        return out

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        """
        Compatibility with policy.greedy and older LSPI code.

        phi_sa(board, move) = features of the afterstate produced by move.
        """
        temporary_move = getattr(board, "temporary_move", None)

        if temporary_move is not None:
            with temporary_move(move):
                return self.phi_afterstate(board)

        state = board.get_state()
        ok = board.make_move(move)

        try:
            if not ok:
                raise ValueError(f"illegal move in phi_sa: {move}")
            return self.phi_afterstate(board)
        finally:
            board.set_state(state)

    # ------------------------------------------------------------------
    # Main v5 feature block
    # ------------------------------------------------------------------

    def _center_features(self, board: Board) -> np.ndarray:
        opening_phase = self._opening_phase(board)

        white_center_pawns = self._weighted_pawn_presence(
            board,
            white=True,
            weights=self.WHITE_CENTER_PAWN_WEIGHTS,
        )
        black_center_pawns = self._weighted_pawn_presence(
            board,
            white=False,
            weights=self.BLACK_CENTER_PAWN_WEIGHTS,
        )

        white_ext_center_pawns = self._pawn_presence_on_squares(
            board,
            white=True,
            squares=self.WHITE_EXTENDED_PAWN_SQUARES,
        )
        black_ext_center_pawns = self._pawn_presence_on_squares(
            board,
            white=False,
            squares=self.BLACK_EXTENDED_PAWN_SQUARES,
        )

        center_pawn_presence_diff = white_center_pawns - black_center_pawns
        extended_center_pawn_presence_diff = white_ext_center_pawns - black_ext_center_pawns

        white_center_occupation = self._occupation_score(
            board,
            white=True,
            squares=self.CORE_CENTER,
        )
        black_center_occupation = self._occupation_score(
            board,
            white=False,
            squares=self.CORE_CENTER,
        )

        center_piece_occupation_diff = white_center_occupation - black_center_occupation

        white_center_control = self._control_score(
            board,
            white=True,
            squares=self.CORE_CENTER,
        )
        black_center_control = self._control_score(
            board,
            white=False,
            squares=self.CORE_CENTER,
        )

        center_control_diff = white_center_control - black_center_control

        white_ext_center_control = self._control_score(
            board,
            white=True,
            squares=self.EXTENDED_CENTER,
        )
        black_ext_center_control = self._control_score(
            board,
            white=False,
            squares=self.EXTENDED_CENTER,
        )

        extended_center_control_diff = white_ext_center_control - black_ext_center_control

        white_minor_center_control = self._piece_type_control_score(
            board,
            white=True,
            squares=self.CORE_CENTER,
            piece_types={p.KNIGHT, p.BISHOP},
        )
        black_minor_center_control = self._piece_type_control_score(
            board,
            white=False,
            squares=self.CORE_CENTER,
            piece_types={p.KNIGHT, p.BISHOP},
        )

        center_minor_control_diff = white_minor_center_control - black_minor_center_control

        white_pawn_center_control = self._piece_type_control_score(
            board,
            white=True,
            squares=self.CORE_CENTER,
            piece_types={p.PAWN},
        )
        black_pawn_center_control = self._piece_type_control_score(
            board,
            white=False,
            squares=self.CORE_CENTER,
            piece_types={p.PAWN},
        )

        center_pawn_control_diff = white_pawn_center_control - black_pawn_center_control

        opening_center_control_diff = opening_phase * center_control_diff
        opening_center_pawn_presence_diff = opening_phase * center_pawn_presence_diff

        white_queen_bad = self._queen_out_before_center(
            board,
            white=True,
            opening_phase=opening_phase,
            own_center_pawns=white_center_pawns,
            own_center_control=white_center_control,
        )
        black_queen_bad = self._queen_out_before_center(
            board,
            white=False,
            opening_phase=opening_phase,
            own_center_pawns=black_center_pawns,
            own_center_control=black_center_control,
        )

        # Positive is good for White:
        # black doing the bad thing is good for White, white doing it is bad.
        queen_out_before_center_diff = black_queen_bad - white_queen_bad

        return np.array(
            [
                center_pawn_presence_diff,
                extended_center_pawn_presence_diff,
                center_piece_occupation_diff,
                center_control_diff,
                extended_center_control_diff,
                center_minor_control_diff,
                center_pawn_control_diff,
                opening_center_control_diff,
                opening_center_pawn_presence_diff,
                queen_out_before_center_diff,
                white_queen_bad,
                black_queen_bad,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Center / opening helpers
    # ------------------------------------------------------------------

    def _opening_phase(self, board: Board) -> float:
        """
        Approximate opening phase in [0, 1].

        Uses both fullmove number and remaining material density.

        This avoids treating artificial endgame FENs with fullmove=1 as
        opening positions.
        """
        fullmove = int(getattr(board, "_fullmove_number", 1))

        # 1.0 in early opening, fading toward 0 around move 16.
        move_phase = (16.0 - fullmove) / 12.0
        move_phase = self._clamp01(move_phase)

        # Count non-king pieces. Initial value is 30.
        nonking_count = 0
        for pc in board.get_board():
            pc = int(pc)
            if pc == p.NONE:
                continue
            if p.piece_type(pc) == p.KING:
                continue
            nonking_count += 1

        material_density = self._clamp01(nonking_count / 30.0)

        # Both should indicate opening-ish.
        return min(move_phase, material_density)

    def _queen_out_before_center(
        self,
        board: Board,
        *,
        white: bool,
        opening_phase: float,
        own_center_pawns: float,
        own_center_control: float,
    ) -> float:
        """
        Penalize early queen sorties when the side has not established
        center presence/control and has not developed enough minor pieces.

        Returns a nonnegative badness value in [0, 1].
        """
        if opening_phase <= 0.0:
            return 0.0

        queen_sq = self._queen_square(board, white=white)
        if queen_sq is None:
            return 0.0

        home = "d1" if white else "d8"
        home_idx = self._idx(board, home)

        if queen_sq == home_idx:
            return 0.0

        minor_dev = self._minor_development_count(board, white=white)

        has_center = own_center_pawns >= 0.50 or own_center_control >= 0.50
        has_development = minor_dev >= 2

        if has_center and has_development:
            return 0.0

        # More severe if both center and development are missing.
        badness = opening_phase

        if not has_center and not has_development:
            badness *= 1.0
        else:
            badness *= 0.65

        return self._clamp01(badness)

    def _minor_development_count(self, board: Board, *, white: bool) -> int:
        """
        Count how many home minor pieces are no longer on their starting squares.

        This treats a traded/captured minor as no longer undeveloped. That is
        acceptable for a compact afterstate feature.
        """
        home = self.WHITE_MINOR_HOME if white else self.BLACK_MINOR_HOME

        undeveloped = 0
        for alg, expected_type in home.items():
            pc = self._piece_at(board, alg)

            if pc == p.NONE:
                continue

            if p.is_white(pc) != white:
                continue

            if p.piece_type(pc) == expected_type:
                undeveloped += 1

        return 4 - undeveloped

    def _weighted_pawn_presence(
        self,
        board: Board,
        *,
        white: bool,
        weights: dict[str, float],
    ) -> float:
        score = 0.0

        for alg, weight in weights.items():
            pc = self._piece_at(board, alg)

            if pc == p.NONE:
                continue

            if p.is_white(pc) == white and p.piece_type(pc) == p.PAWN:
                score += weight

        # d/e pawn duo gives 2.0 -> 1.0.
        return self._clamp01(score / 2.0)

    def _pawn_presence_on_squares(
        self,
        board: Board,
        *,
        white: bool,
        squares: tuple[str, ...],
    ) -> float:
        count = 0

        for alg in squares:
            pc = self._piece_at(board, alg)

            if pc == p.NONE:
                continue

            if p.is_white(pc) == white and p.piece_type(pc) == p.PAWN:
                count += 1

        return self._clamp01(count / max(1, len(squares)))

    def _occupation_score(
        self,
        board: Board,
        *,
        white: bool,
        squares: tuple[str, ...],
    ) -> float:
        """
        Score occupation of core center.

        Pawns/minors matter more than queen/rook occupation, because early queen
        centralization should not be strongly rewarded.
        """
        score = 0.0

        for alg in squares:
            pc = self._piece_at(board, alg)

            if pc == p.NONE:
                continue

            if p.is_white(pc) != white:
                continue

            typ = p.piece_type(pc)

            if typ == p.PAWN:
                score += 1.0
            elif typ in (p.KNIGHT, p.BISHOP):
                score += 0.75
            elif typ == p.KING:
                score += 0.0
            else:
                score += 0.25

        return self._clamp01(score / len(squares))

    def _control_score(
        self,
        board: Board,
        *,
        white: bool,
        squares: tuple[str, ...],
    ) -> float:
        """
        Fraction of target squares controlled/attacked by the side.
        """
        if not squares:
            return 0.0

        controlled = 0

        for alg in squares:
            sq = self._idx(board, alg)

            try:
                if board.is_square_attacked(sq, by_white=white):
                    controlled += 1
            except TypeError:
                # Fallback if the method does not accept keyword args.
                if board.is_square_attacked(sq, white):
                    controlled += 1

        return controlled / len(squares)

    def _piece_type_control_score(
        self,
        board: Board,
        *,
        white: bool,
        squares: tuple[str, ...],
        piece_types: set[int],
    ) -> float:
        """
        Fraction of target squares controlled by at least one piece whose type
        is in piece_types.
        """
        if not squares:
            return 0.0

        controlled = 0

        for dst_alg in squares:
            dst_idx = self._idx(board, dst_alg)

            if self._is_square_attacked_by_piece_types(
                board,
                dst_idx=dst_idx,
                by_white=white,
                piece_types=piece_types,
            ):
                controlled += 1

        return controlled / len(squares)

    # ------------------------------------------------------------------
    # Low-level board geometry helpers
    # ------------------------------------------------------------------

    def _queen_square(self, board: Board, *, white: bool) -> int | None:
        for idx, pc in enumerate(board.get_board()):
            pc = int(pc)

            if pc == p.NONE:
                continue

            if p.is_white(pc) == white and p.piece_type(pc) == p.QUEEN:
                return idx

        return None

    def _piece_at(self, board: Board, alg: str) -> int:
        return int(board.get_board()[self._idx(board, alg)])

    def _idx(self, board: Board, alg: str) -> int:
        return board.algebraic_to_idx(alg)

    def _coord(self, board: Board, idx: int) -> tuple[int, int]:
        alg = board.idx_to_algebraic(idx)
        file_i = ord(alg[0]) - ord("a")
        rank_i = int(alg[1:]) - 1
        return file_i, rank_i

    def _idx_from_coord(self, board: Board, file_i: int, rank_i: int) -> int | None:
        if not (0 <= file_i < 8 and 0 <= rank_i < 8):
            return None

        alg = f"{chr(ord('a') + file_i)}{rank_i + 1}"
        return board.algebraic_to_idx(alg)

    def _is_square_attacked_by_piece_types(
        self,
        board: Board,
        *,
        dst_idx: int,
        by_white: bool,
        piece_types: set[int],
    ) -> bool:
        arr = board.get_board()

        for src_idx, pc in enumerate(arr):
            pc = int(pc)

            if pc == p.NONE:
                continue

            if p.is_white(pc) != by_white:
                continue

            typ = p.piece_type(pc)

            if typ not in piece_types:
                continue

            if self._piece_attacks_square(board, src_idx=src_idx, dst_idx=dst_idx, piece=pc):
                return True

        return False

    def _piece_attacks_square(
        self,
        board: Board,
        *,
        src_idx: int,
        dst_idx: int,
        piece: int,
    ) -> bool:
        src_f, src_r = self._coord(board, src_idx)
        dst_f, dst_r = self._coord(board, dst_idx)

        df = dst_f - src_f
        dr = dst_r - src_r

        typ = p.piece_type(piece)
        white = p.is_white(piece)

        if typ == p.PAWN:
            direction = 1 if white else -1
            return dr == direction and abs(df) == 1

        if typ == p.KNIGHT:
            return (abs(df), abs(dr)) in ((1, 2), (2, 1))

        if typ == p.KING:
            return max(abs(df), abs(dr)) == 1

        if typ == p.BISHOP:
            if abs(df) != abs(dr):
                return False
            return self._ray_clear(board, src_f, src_r, dst_f, dst_r)

        if typ == p.ROOK:
            if df != 0 and dr != 0:
                return False
            return self._ray_clear(board, src_f, src_r, dst_f, dst_r)

        if typ == p.QUEEN:
            diagonal = abs(df) == abs(dr)
            straight = df == 0 or dr == 0

            if not diagonal and not straight:
                return False

            return self._ray_clear(board, src_f, src_r, dst_f, dst_r)

        return False

    def _ray_clear(
        self,
        board: Board,
        src_f: int,
        src_r: int,
        dst_f: int,
        dst_r: int,
    ) -> bool:
        step_f = self._sign(dst_f - src_f)
        step_r = self._sign(dst_r - src_r)

        f = src_f + step_f
        r = src_r + step_r

        while (f, r) != (dst_f, dst_r):
            idx = self._idx_from_coord(board, f, r)

            if idx is None:
                return False

            if int(board.get_board()[idx]) != p.NONE:
                return False

            f += step_f
            r += step_r

        return True

    def _sign(self, x: int) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))