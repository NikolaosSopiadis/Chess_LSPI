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


class V6AttackMapFeatures:
    """
    v6_attackmap = v4_slim + attack-count / tension features.

    Sign convention:
      Positive feature values are good for White.
      Negative feature values are good for Black.

    Base:
      0..36  = v4_slim

    Added:
      37 attack_coverage_diff
      38 attack_volume_diff
      39 minor_attack_volume_diff
      40 pawn_attack_volume_diff
      41 nonqueen_attack_volume_diff
      42 center_attack_volume_diff
      43 extended_center_attack_volume_diff
      44 own_piece_defense_volume_diff
      45 king_zone_attack_volume_diff
      46 king_zone_multi_attack_diff
      47 king_zone_pawn_attack_diff
      48 king_zone_minor_attack_diff
      49 king_zone_net_attack_diff
      50 queen_pressure_diff
      51 queen_pressure_by_minor_or_pawn_diff
      52 queen_net_attack_defense_diff
      53 loose_piece_pressure_diff
      54 overloaded_piece_pressure_diff
      55 multi_attacked_material_diff
      56 tempo_liability_diff
    """

    spec = _FeatureSpec(version="v6_attackmap", dim=57)

    CORE_CENTER = ("d4", "e4", "d5", "e5")

    EXTENDED_CENTER = (
        "c3", "d3", "e3", "f3",
        "c4", "d4", "e4", "f4",
        "c5", "d5", "e5", "f5",
        "c6", "d6", "e6", "f6",
    )

    PAWN_TYPES = frozenset({p.PAWN})
    MINOR_TYPES = frozenset({p.KNIGHT, p.BISHOP})
    PAWN_OR_MINOR_TYPES = frozenset({p.PAWN, p.KNIGHT, p.BISHOP})
    NONQUEEN_ATTACK_TYPES = frozenset({p.PAWN, p.KNIGHT, p.BISHOP, p.ROOK, p.KING})

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

        self._core_center_idx = tuple(self._alg_to_idx_static(sq) for sq in self.CORE_CENTER)
        self._extended_center_idx = tuple(self._alg_to_idx_static(sq) for sq in self.EXTENDED_CENTER)

    def phi_afterstate(self, board: Board) -> np.ndarray:
        base = np.asarray(self._base.phi_afterstate(board), dtype=np.float64)

        if base.shape[0] != 37:
            raise ValueError(
                f"V6AttackMapFeatures expected v4_slim dim 37, got {base.shape[0]}"
            )

        extra = self._attackmap_features(board)

        out = np.concatenate([base, extra]).astype(np.float64, copy=False)

        if out.shape[0] != self.spec.dim:
            raise ValueError(f"v6_attackmap produced dim {out.shape[0]}, expected {self.spec.dim}")

        return out

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        """
        Compatibility with older policy/search code.

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
    # Main feature block
    # ------------------------------------------------------------------

    def _attackmap_features(self, board: Board) -> np.ndarray:
        white_attacks, black_attacks = board.attack_maps()

        white_minor_attacks, black_minor_attacks = board.attack_maps(
            piece_types=self.MINOR_TYPES
        )
        white_pawn_attacks, black_pawn_attacks = board.attack_maps(
            piece_types=self.PAWN_TYPES
        )
        white_pawn_minor_attacks, black_pawn_minor_attacks = board.attack_maps(
            piece_types=self.PAWN_OR_MINOR_TYPES
        )
        white_nonqueen_attacks, black_nonqueen_attacks = board.attack_maps(
            piece_types=self.NONQUEEN_ATTACK_TYPES
        )

        arr = board.get_board()

        white_king_sq = self._king_square(board, white=True)
        black_king_sq = self._king_square(board, white=False)

        white_king_zone = self._king_zone(white_king_sq)
        black_king_zone = self._king_zone(black_king_sq)

        # ------------------------------------------------------------------
        # Global attack-map activity
        # ------------------------------------------------------------------

        attack_coverage_diff = self._diff_norm(
            self._coverage(white_attacks),
            self._coverage(black_attacks),
            64.0,
        )

        attack_volume_diff = self._diff_norm(
            sum(white_attacks),
            sum(black_attacks),
            64.0,
        )

        minor_attack_volume_diff = self._diff_norm(
            sum(white_minor_attacks),
            sum(black_minor_attacks),
            32.0,
        )

        pawn_attack_volume_diff = self._diff_norm(
            sum(white_pawn_attacks),
            sum(black_pawn_attacks),
            16.0,
        )

        nonqueen_attack_volume_diff = self._diff_norm(
            sum(white_nonqueen_attacks),
            sum(black_nonqueen_attacks),
            64.0,
        )

        # ------------------------------------------------------------------
        # Center attack pressure
        # ------------------------------------------------------------------

        center_attack_volume_diff = self._diff_norm(
            self._sum_on(white_attacks, self._core_center_idx),
            self._sum_on(black_attacks, self._core_center_idx),
            12.0,
        )

        extended_center_attack_volume_diff = self._diff_norm(
            self._sum_on(white_attacks, self._extended_center_idx),
            self._sum_on(black_attacks, self._extended_center_idx),
            32.0,
        )

        # ------------------------------------------------------------------
        # Own-piece defense volume
        # ------------------------------------------------------------------

        white_piece_defense = self._own_piece_defense_volume(
            arr,
            own_white=True,
            own_attacks=white_attacks,
        )
        black_piece_defense = self._own_piece_defense_volume(
            arr,
            own_white=False,
            own_attacks=black_attacks,
        )

        own_piece_defense_volume_diff = self._diff_norm(
            white_piece_defense,
            black_piece_defense,
            32.0,
        )

        # ------------------------------------------------------------------
        # King-zone pressure
        # ------------------------------------------------------------------

        white_pressure_on_black_king = self._sum_on(white_attacks, black_king_zone)
        black_pressure_on_white_king = self._sum_on(black_attacks, white_king_zone)

        king_zone_attack_volume_diff = self._diff_norm(
            white_pressure_on_black_king,
            black_pressure_on_white_king,
            16.0,
        )

        king_zone_multi_attack_diff = self._diff_norm(
            self._multi_attacked_count(white_attacks, black_king_zone),
            self._multi_attacked_count(black_attacks, white_king_zone),
            8.0,
        )

        king_zone_pawn_attack_diff = self._diff_norm(
            self._sum_on(white_pawn_attacks, black_king_zone),
            self._sum_on(black_pawn_attacks, white_king_zone),
            8.0,
        )

        king_zone_minor_attack_diff = self._diff_norm(
            self._sum_on(white_minor_attacks, black_king_zone),
            self._sum_on(black_minor_attacks, white_king_zone),
            8.0,
        )

        white_net_on_black_king = self._net_zone_pressure(
            attackers=white_attacks,
            defenders=black_attacks,
            zone=black_king_zone,
        )
        black_net_on_white_king = self._net_zone_pressure(
            attackers=black_attacks,
            defenders=white_attacks,
            zone=white_king_zone,
        )

        king_zone_net_attack_diff = self._diff_norm(
            white_net_on_black_king,
            black_net_on_white_king,
            10.0,
        )

        # ------------------------------------------------------------------
        # Queen pressure
        # ------------------------------------------------------------------

        white_queen_sq = self._queen_square(board, white=True)
        black_queen_sq = self._queen_square(board, white=False)

        queen_pressure_diff = self._queen_pressure_diff(
            white_queen_sq=white_queen_sq,
            black_queen_sq=black_queen_sq,
            white_attacks=white_attacks,
            black_attacks=black_attacks,
            divisor=4.0,
        )

        queen_pressure_by_minor_or_pawn_diff = self._queen_pressure_diff(
            white_queen_sq=white_queen_sq,
            black_queen_sq=black_queen_sq,
            white_attacks=white_pawn_minor_attacks,
            black_attacks=black_pawn_minor_attacks,
            divisor=3.0,
        )

        queen_net_attack_defense_diff = self._queen_net_attack_defense_diff(
            white_queen_sq=white_queen_sq,
            black_queen_sq=black_queen_sq,
            white_attacks=white_attacks,
            black_attacks=black_attacks,
            divisor=4.0,
        )

        # ------------------------------------------------------------------
        # General attacked/defended material tension
        # ------------------------------------------------------------------

        loose_piece_pressure_diff = self._loose_piece_pressure_diff(
            arr,
            white_attacks=white_attacks,
            black_attacks=black_attacks,
        )

        overloaded_piece_pressure_diff = self._overloaded_piece_pressure_diff(
            arr,
            white_attacks=white_attacks,
            black_attacks=black_attacks,
        )

        multi_attacked_material_diff = self._multi_attacked_material_diff(
            arr,
            white_attacks=white_attacks,
            black_attacks=black_attacks,
        )

        tempo_liability_diff = self._tempo_liability_diff(
            arr,
            white_pawn_minor_attacks=white_pawn_minor_attacks,
            black_pawn_minor_attacks=black_pawn_minor_attacks,
            white_attacks=white_attacks,
            black_attacks=black_attacks,
        )

        return np.array(
            [
                attack_coverage_diff,
                attack_volume_diff,
                minor_attack_volume_diff,
                pawn_attack_volume_diff,
                nonqueen_attack_volume_diff,
                center_attack_volume_diff,
                extended_center_attack_volume_diff,
                own_piece_defense_volume_diff,
                king_zone_attack_volume_diff,
                king_zone_multi_attack_diff,
                king_zone_pawn_attack_diff,
                king_zone_minor_attack_diff,
                king_zone_net_attack_diff,
                queen_pressure_diff,
                queen_pressure_by_minor_or_pawn_diff,
                queen_net_attack_defense_diff,
                loose_piece_pressure_diff,
                overloaded_piece_pressure_diff,
                multi_attacked_material_diff,
                tempo_liability_diff,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Attack-map helper features
    # ------------------------------------------------------------------

    def _coverage(self, attack_map: tuple[int, ...]) -> int:
        return sum(1 for x in attack_map if x > 0)

    def _sum_on(self, attack_map: tuple[int, ...], squares: tuple[int, ...]) -> int:
        return sum(int(attack_map[sq]) for sq in squares)

    def _multi_attacked_count(self, attack_map: tuple[int, ...], squares: tuple[int, ...]) -> int:
        return sum(1 for sq in squares if int(attack_map[sq]) >= 2)

    def _own_piece_defense_volume(
        self,
        arr,
        *,
        own_white: bool,
        own_attacks: tuple[int, ...],
    ) -> float:
        total = 0.0

        for sq, pc_raw in enumerate(arr):
            pc = int(pc_raw)

            if pc == p.NONE:
                continue

            if p.is_white(pc) != own_white:
                continue

            if p.piece_type(pc) == p.KING:
                continue

            total += min(3, int(own_attacks[sq]))

        return total

    def _net_zone_pressure(
        self,
        *,
        attackers: tuple[int, ...],
        defenders: tuple[int, ...],
        zone: tuple[int, ...],
    ) -> float:
        total = 0.0

        for sq in zone:
            total += max(0, int(attackers[sq]) - int(defenders[sq]))

        return total

    def _queen_pressure_diff(
        self,
        *,
        white_queen_sq: int | None,
        black_queen_sq: int | None,
        white_attacks: tuple[int, ...],
        black_attacks: tuple[int, ...],
        divisor: float,
    ) -> float:
        # Positive = White pressures Black queen more than Black pressures White queen.
        white_pressure_on_black_queen = 0 if black_queen_sq is None else int(white_attacks[black_queen_sq])
        black_pressure_on_white_queen = 0 if white_queen_sq is None else int(black_attacks[white_queen_sq])

        return self._diff_norm(
            white_pressure_on_black_queen,
            black_pressure_on_white_queen,
            divisor,
        )

    def _queen_net_attack_defense_diff(
        self,
        *,
        white_queen_sq: int | None,
        black_queen_sq: int | None,
        white_attacks: tuple[int, ...],
        black_attacks: tuple[int, ...],
        divisor: float,
    ) -> float:
        # For Black queen:
        #   attackers = white_attacks
        #   defenders = black_attacks
        if black_queen_sq is None:
            black_queen_net_pressure = 0
        else:
            black_queen_net_pressure = max(
                0,
                int(white_attacks[black_queen_sq]) - int(black_attacks[black_queen_sq]),
            )

        # For White queen:
        #   attackers = black_attacks
        #   defenders = white_attacks
        if white_queen_sq is None:
            white_queen_net_pressure = 0
        else:
            white_queen_net_pressure = max(
                0,
                int(black_attacks[white_queen_sq]) - int(white_attacks[white_queen_sq]),
            )

        return self._diff_norm(
            black_queen_net_pressure,
            white_queen_net_pressure,
            divisor,
        )

    def _loose_piece_pressure_diff(
        self,
        arr,
        *,
        white_attacks: tuple[int, ...],
        black_attacks: tuple[int, ...],
    ) -> float:
        """
        Material value of enemy loose pieces attacked by us minus own loose
        pieces attacked by enemy.

        Loose = attacked by enemy and defended by own side zero times.
        """
        white_score = 0.0
        black_score = 0.0

        for sq, pc_raw in enumerate(arr):
            pc = int(pc_raw)

            if pc == p.NONE:
                continue

            typ = p.piece_type(pc)

            if typ == p.KING:
                continue

            value = self.PIECE_VALUE.get(typ, 0.0)

            if p.is_white(pc):
                enemy_attacks = int(black_attacks[sq])
                own_defenders = int(white_attacks[sq])

                if enemy_attacks > 0 and own_defenders == 0:
                    black_score += value
            else:
                enemy_attacks = int(white_attacks[sq])
                own_defenders = int(black_attacks[sq])

                if enemy_attacks > 0 and own_defenders == 0:
                    white_score += value

        return self._diff_norm(white_score, black_score, 20.0)

    def _overloaded_piece_pressure_diff(
        self,
        arr,
        *,
        white_attacks: tuple[int, ...],
        black_attacks: tuple[int, ...],
    ) -> float:
        """
        Material pressure on pieces with more attackers than defenders.

        Positive means White has more pressure on overloaded Black pieces.
        """
        white_score = 0.0
        black_score = 0.0

        for sq, pc_raw in enumerate(arr):
            pc = int(pc_raw)

            if pc == p.NONE:
                continue

            typ = p.piece_type(pc)

            if typ == p.KING:
                continue

            value = self.PIECE_VALUE.get(typ, 0.0)

            if p.is_white(pc):
                attackers = int(black_attacks[sq])
                defenders = int(white_attacks[sq])
                excess = max(0, attackers - defenders)

                if excess > 0:
                    black_score += value * min(3, excess)
            else:
                attackers = int(white_attacks[sq])
                defenders = int(black_attacks[sq])
                excess = max(0, attackers - defenders)

                if excess > 0:
                    white_score += value * min(3, excess)

        return self._diff_norm(white_score, black_score, 25.0)

    def _multi_attacked_material_diff(
        self,
        arr,
        *,
        white_attacks: tuple[int, ...],
        black_attacks: tuple[int, ...],
    ) -> float:
        """
        Material value of enemy pieces attacked at least twice minus own pieces
        attacked at least twice.
        """
        white_score = 0.0
        black_score = 0.0

        for sq, pc_raw in enumerate(arr):
            pc = int(pc_raw)

            if pc == p.NONE:
                continue

            typ = p.piece_type(pc)

            if typ == p.KING:
                continue

            value = self.PIECE_VALUE.get(typ, 0.0)

            if p.is_white(pc):
                if int(black_attacks[sq]) >= 2:
                    black_score += value
            else:
                if int(white_attacks[sq]) >= 2:
                    white_score += value

        return self._diff_norm(white_score, black_score, 20.0)

    def _tempo_liability_diff(
        self,
        arr,
        *,
        white_pawn_minor_attacks: tuple[int, ...],
        black_pawn_minor_attacks: tuple[int, ...],
        white_attacks: tuple[int, ...],
        black_attacks: tuple[int, ...],
    ) -> float:
        """
        Tempo liability from valuable pieces being attackable by pawns/minors.

        Positive means Black has more tempo-liable pieces.
        Negative means White has more tempo-liable pieces.

        This is meant to detect things like an early queen becoming a target,
        without explicitly hardcoding opening rules.
        """
        white_liability = 0.0
        black_liability = 0.0

        for sq, pc_raw in enumerate(arr):
            pc = int(pc_raw)

            if pc == p.NONE:
                continue

            typ = p.piece_type(pc)

            if typ in (p.KING, p.PAWN):
                continue

            base = {
                p.KNIGHT: 0.35,
                p.BISHOP: 0.35,
                p.ROOK: 0.65,
                p.QUEEN: 1.25,
            }.get(typ, 0.0)

            if base <= 0.0:
                continue

            if p.is_white(pc):
                cheap_attackers = int(black_pawn_minor_attacks[sq])
                own_defenders = int(white_attacks[sq])

                if cheap_attackers <= 0:
                    continue

                liability = base * cheap_attackers

                # If defended enough, it is still a tempo target, but less severe.
                if own_defenders >= cheap_attackers:
                    liability *= 0.5

                white_liability += liability

            else:
                cheap_attackers = int(white_pawn_minor_attacks[sq])
                own_defenders = int(black_attacks[sq])

                if cheap_attackers <= 0:
                    continue

                liability = base * cheap_attackers

                if own_defenders >= cheap_attackers:
                    liability *= 0.5

                black_liability += liability

        return self._diff_norm(black_liability, white_liability, 6.0)

    # ------------------------------------------------------------------
    # Board geometry helpers
    # ------------------------------------------------------------------

    def _king_square(self, board: Board, *, white: bool) -> int:
        attr = "_white_king_sq" if white else "_black_king_sq"
        sq = int(getattr(board, attr, -1))

        if 0 <= sq < 64:
            return sq

        target = p.WHITE_KING if white else p.BLACK_KING

        for idx, pc in enumerate(board.get_board()):
            if int(pc) == target:
                return idx

        return -1

    def _queen_square(self, board: Board, *, white: bool) -> int | None:
        target = p.WHITE_QUEEN if white else p.BLACK_QUEEN

        for idx, pc in enumerate(board.get_board()):
            if int(pc) == target:
                return idx

        return None

    def _king_zone(self, king_sq: int) -> tuple[int, ...]:
        """
        King square + adjacent squares.
        """
        if not (0 <= king_sq < 64):
            return ()

        f0 = king_sq & 7
        r0 = king_sq >> 3

        out: list[int] = []

        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                f = f0 + df
                r = r0 + dr

                if 0 <= f < 8 and 0 <= r < 8:
                    out.append(f + (r << 3))

        return tuple(out)

    def _alg_to_idx_static(self, sq: str) -> int:
        f = ord(sq[0]) - ord("a")
        r = int(sq[1]) - 1

        if not (0 <= f < 8 and 0 <= r < 8):
            raise ValueError(f"bad square: {sq}")

        return f + (r << 3)

    def _diff_norm(self, white_value: float, black_value: float, divisor: float) -> float:
        if divisor <= 0:
            raise ValueError("divisor must be positive")

        return self._clamp(float(white_value - black_value) / divisor, -1.0, 1.0)

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))