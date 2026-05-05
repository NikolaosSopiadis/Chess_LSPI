from __future__ import annotations

import numpy as np

from chess_core.board import Board
from chess_core.piece import Piece as p
from chess_core.move import Move

from chess_rl.features.base import FeatureSpec
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


AHEAD_THRESHOLD = 0.30


class V4SlimFeatures:
    """
    Slim v4 feature set.

    Goals:
      - remove redundant v3 white/black/diff triples,
      - remove weak castling-right features,
      - remove expensive legal-mobility features,
      - add explicit king-safety / castling / development features.

    Feature convention:
      Most signed difference features are white-perspective:
        positive = good for White
        negative = good for Black
    """

    spec = FeatureSpec(
        name="features",
        version="v4_slim",
        dim=37,
    )

    def __init__(self) -> None:
        self._sq_cache: dict[str, int] = {}
        self._coord_cache: dict[int, tuple[int, int]] = {}

    def phi_afterstate(self, board: Board) -> np.ndarray:
        x = np.zeros(self.spec.dim, dtype=np.float32)

        material = self._piece_count_features(board)

        white_mobility = self._pseudo_mobility(board, white=True)
        black_mobility = self._pseudo_mobility(board, white=False)

        white_attacks_black = self._attacked_material(board, attacker_white=True)
        black_attacks_white = self._attacked_material(board, attacker_white=False)

        white_hanging = self._hanging_material(board, white=True)
        black_hanging = self._hanging_material(board, white=False)

        white_king_danger = self._king_zone_attacked_score(board, white=True)
        black_king_danger = self._king_zone_attacked_score(board, white=False)

        white_pawn_adv = self._pawn_advancement_score(board, white=True)
        black_pawn_adv = self._pawn_advancement_score(board, white=False)

        white_passed = self._passed_pawn_score(board, white=True)
        black_passed = self._passed_pawn_score(board, white=False)

        white_promo = self._promotion_pressure_score(board, white=True)
        black_promo = self._promotion_pressure_score(board, white=False)

        white_castled = float(self._is_castled(board, white=True))
        black_castled = float(self._is_castled(board, white=False))

        white_home = float(self._king_on_home_square(board, white=True))
        black_home = float(self._king_on_home_square(board, white=False))

        white_walked_uncastled = float((not white_castled) and (not white_home))
        black_walked_uncastled = float((not black_castled) and (not black_home))

        white_pawn_shield = self._pawn_shield_score(board, white=True)
        black_pawn_shield = self._pawn_shield_score(board, white=False)

        white_open_files = self._open_files_near_king_score(board, white=True)
        black_open_files = self._open_files_near_king_score(board, white=False)

        white_dev = self._minor_development_score(board, white=True)
        black_dev = self._minor_development_score(board, white=False)

        white_queen_dev = float(self._queen_developed(board, white=True))
        black_queen_dev = float(self._queen_developed(board, white=False))

        done, reason = board.game_end_state()
        mat = float(material_potential(board))

        is_draw = done and reason in {
            "stalemate",
            "threefold repetition",
            "fifty-move rule",
            "insufficient material",
        }

        white_ahead = mat > AHEAD_THRESHOLD
        black_ahead = mat < -AHEAD_THRESHOLD

        try:
            rep_count = int(board.current_repetition_count())
        except Exception:
            rep_count = 0

        repeat_risk = rep_count >= 2

        halfmove = int(getattr(board, "_halfmove_clock", 0))
        halfmove_pressure = 0.0
        if halfmove >= 80:
            halfmove_pressure = (halfmove - 80) / 20.0
            halfmove_pressure = max(0.0, min(1.0, halfmove_pressure))

        # 0: bias
        x[0] = 1.0

        # 1-5: normalized piece-count diffs
        x[1] = material["pawn_diff"]
        x[2] = material["knight_diff"]
        x[3] = material["bishop_diff"]
        x[4] = material["rook_diff"]
        x[5] = material["queen_diff"]

        # 6-8: side/check state
        x[6] = 1.0 if board.get_is_white_to_move() else -1.0
        x[7] = float(board.in_check(True))
        x[8] = float(board.in_check(False))

        # 9-15: tactical/positional signed diffs
        x[9] = (white_mobility - black_mobility) / 64.0

        # Positive means White attacks more Black material.
        x[10] = white_attacks_black - black_attacks_white

        # Positive means more Black material is hanging than White material.
        x[11] = black_hanging - white_hanging

        # Positive means Black king is under more pressure than White king.
        x[12] = black_king_danger - white_king_danger

        x[13] = white_pawn_adv - black_pawn_adv
        x[14] = white_passed - black_passed
        x[15] = white_promo - black_promo

        # 16-17: terminal checkmate
        x[16] = float(done and reason == "checkmate" and not board.get_is_white_to_move())
        x[17] = float(done and reason == "checkmate" and board.get_is_white_to_move())

        # 18-23: draw/conversion pressure
        x[18] = float(is_draw and white_ahead)
        x[19] = float(is_draw and black_ahead)

        x[20] = float(repeat_risk and white_ahead)
        x[21] = float(repeat_risk and black_ahead)

        x[22] = halfmove_pressure if white_ahead else 0.0
        x[23] = halfmove_pressure if black_ahead else 0.0

        # 24-30: king safety / development diffs
        x[24] = white_castled - black_castled

        # Positive means Black has walked the king uncastled more than White.
        x[25] = black_walked_uncastled - white_walked_uncastled

        x[26] = white_pawn_shield - black_pawn_shield

        # Positive means Black king zone is more attacked than White king zone.
        x[27] = black_king_danger - white_king_danger

        # Positive means Black has more open files near king.
        x[28] = black_open_files - white_open_files

        x[29] = white_dev - black_dev

        # Early queen development is often risky.
        # Positive means Black queen is developed more than White queen.
        x[30] = black_queen_dev - white_queen_dev

        # 31-36: individual king-safety diagnostics
        x[31] = white_castled
        x[32] = black_castled
        x[33] = white_pawn_shield
        x[34] = black_pawn_shield
        x[35] = white_king_danger
        x[36] = black_king_danger

        return x

    def __call__(self, board: Board) -> np.ndarray:
        return self.phi_afterstate(board)

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        """
        Compatibility with older greedy policy code.

        phi_sa(board, move) = features of the afterstate produced by move.
        """
        with board.temporary_move(move):
            return self.phi_afterstate(board)

    def _piece_count_features(self, board: Board) -> dict[str, float]:
        counts = {
            "wp": 0,
            "wn": 0,
            "wb": 0,
            "wr": 0,
            "wq": 0,
            "bp": 0,
            "bn": 0,
            "bb": 0,
            "br": 0,
            "bq": 0,
        }

        for piece in board.get_board():
            piece = int(piece)

            if piece == p.NONE:
                continue

            white = p.is_white(piece)
            t = p.piece_type(piece)

            prefix = "w" if white else "b"

            if t == p.PAWN:
                counts[prefix + "p"] += 1
            elif t == p.KNIGHT:
                counts[prefix + "n"] += 1
            elif t == p.BISHOP:
                counts[prefix + "b"] += 1
            elif t == p.ROOK:
                counts[prefix + "r"] += 1
            elif t == p.QUEEN:
                counts[prefix + "q"] += 1

        return {
            "pawn_diff": (counts["wp"] - counts["bp"]) / 8.0,
            "knight_diff": (counts["wn"] - counts["bn"]) / 2.0,
            "bishop_diff": (counts["wb"] - counts["bb"]) / 2.0,
            "rook_diff": (counts["wr"] - counts["br"]) / 2.0,
            "queen_diff": float(counts["wq"] - counts["bq"]),
        }

    def _pseudo_mobility(self, board: Board, *, white: bool) -> int:
        old_turn = getattr(board, "_is_white_to_move", None)
        old_ep = getattr(board, "_en_passant_target", None)

        try:
            if old_turn is not None:
                board._is_white_to_move = white

            # En-passant is only valid for the real side-to-move.
            if old_turn is not None and white != old_turn:
                board._en_passant_target = None

            total = 0

            for src, piece in enumerate(board.get_board()):
                piece = int(piece)

                if piece == p.NONE:
                    continue

                if p.is_white(piece) != white:
                    continue

                total += len(board.get_pseudolegal_moves(src))

            return total

        finally:
            if old_turn is not None:
                board._is_white_to_move = old_turn
            if old_ep is not None or hasattr(board, "_en_passant_target"):
                board._en_passant_target = old_ep

    def _piece_value(self, piece: int) -> float:
        if piece == p.NONE:
            return 0.0

        t = p.piece_type(piece)

        if t == p.PAWN:
            return 0.10
        if t == p.KNIGHT:
            return 0.32
        if t == p.BISHOP:
            return 0.33
        if t == p.ROOK:
            return 0.50
        if t == p.QUEEN:
            return 0.90

        return 0.0

    def _attacked_material(self, board: Board, *, attacker_white: bool) -> float:
        total = 0.0

        for sq, piece in enumerate(board.get_board()):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) == attacker_white:
                continue

            if board.is_square_attacked(sq, by_white=attacker_white):
                total += self._piece_value(piece)

        return total

    def _hanging_material(self, board: Board, *, white: bool) -> float:
        """
        Material of this side's pieces that are attacked and not defended.
        """
        enemy_white = not white
        total = 0.0

        for sq, piece in enumerate(board.get_board()):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) != white:
                continue

            if not board.is_square_attacked(sq, by_white=enemy_white):
                continue

            if board.is_square_attacked(sq, by_white=white):
                continue

            total += self._piece_value(piece)

        return total

    def _idx(self, board: Board, square: str) -> int:
        cached = self._sq_cache.get(square)
        if cached is not None:
            return cached

        idx = board.algebraic_to_idx(square)
        self._sq_cache[square] = idx
        return idx

    def _piece_at(self, board: Board, square: str) -> int:
        return int(board.get_board()[self._idx(board, square)])

    def _has_piece(self, board: Board, square: str, *, white: bool, piece_type: int) -> bool:
        piece = self._piece_at(board, square)

        if piece == p.NONE:
            return False

        return p.is_white(piece) == white and p.piece_type(piece) == piece_type

    def _coord(self, board: Board, sq: int) -> tuple[int, int]:
        cached = self._coord_cache.get(sq)
        if cached is not None:
            return cached

        alg = board.idx_to_algebraic(sq)
        file_idx = ord(alg[0]) - ord("a")
        rank = int(alg[1])

        coord = (file_idx, rank)
        self._coord_cache[sq] = coord
        return coord

    def _square_from_file_rank(self, board: Board, file_idx: int, rank: int) -> int | None:
        if file_idx < 0 or file_idx > 7:
            return None

        if rank < 1 or rank > 8:
            return None

        file_char = chr(ord("a") + file_idx)
        return self._idx(board, f"{file_char}{rank}")

    def _king_square(self, board: Board, *, white: bool) -> int | None:
        for sq, piece in enumerate(board.get_board()):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) == white and p.piece_type(piece) == p.KING:
                return sq

        return None

    def _is_castled(self, board: Board, *, white: bool) -> bool:
        king_sq = self._king_square(board, white=white)

        if king_sq is None:
            return False

        if white:
            return king_sq in {
                self._idx(board, "g1"),
                self._idx(board, "c1"),
            }

        return king_sq in {
            self._idx(board, "g8"),
            self._idx(board, "c8"),
        }

    def _king_on_home_square(self, board: Board, *, white: bool) -> bool:
        if white:
            return self._has_piece(board, "e1", white=True, piece_type=p.KING)

        return self._has_piece(board, "e8", white=False, piece_type=p.KING)

    def _is_own_pawn(self, board: Board, sq: int, *, white: bool) -> bool:
        piece = int(board.get_board()[sq])

        if piece == p.NONE:
            return False

        return p.is_white(piece) == white and p.piece_type(piece) == p.PAWN

    def _pawn_shield_score(self, board: Board, *, white: bool) -> float:
        king_sq = self._king_square(board, white=white)

        if king_sq is None:
            return 0.0

        king_file, king_rank = self._coord(board, king_sq)

        files = [king_file - 1, king_file, king_file + 1]

        if white:
            ranks = [king_rank + 1, king_rank + 2]
        else:
            ranks = [king_rank - 1, king_rank - 2]

        total = 0
        pawns = 0

        for f in files:
            for r in ranks:
                sq = self._square_from_file_rank(board, f, r)
                if sq is None:
                    continue

                total += 1

                if self._is_own_pawn(board, sq, white=white):
                    pawns += 1

        if total == 0:
            return 0.0

        return pawns / total

    def _king_zone(self, board: Board, *, white: bool) -> list[int]:
        king_sq = self._king_square(board, white=white)

        if king_sq is None:
            return []

        king_file, king_rank = self._coord(board, king_sq)

        zone: list[int] = []

        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                sq = self._square_from_file_rank(
                    board,
                    king_file + df,
                    king_rank + dr,
                )

                if sq is not None:
                    zone.append(sq)

        return zone

    def _king_zone_attacked_score(self, board: Board, *, white: bool) -> float:
        zone = self._king_zone(board, white=white)

        if not zone:
            return 0.0

        enemy_is_white = not white
        attacked = 0

        for sq in zone:
            if board.is_square_attacked(sq, by_white=enemy_is_white):
                attacked += 1

        return attacked / len(zone)

    def _open_files_near_king_score(self, board: Board, *, white: bool) -> float:
        king_sq = self._king_square(board, white=white)

        if king_sq is None:
            return 0.0

        king_file, _king_rank = self._coord(board, king_sq)

        files = [king_file - 1, king_file, king_file + 1]

        total_files = 0
        files_without_own_pawn = 0

        for f in files:
            if f < 0 or f > 7:
                continue

            total_files += 1
            has_own_pawn = False

            for rank in range(1, 9):
                sq = self._square_from_file_rank(board, f, rank)
                if sq is None:
                    continue

                if self._is_own_pawn(board, sq, white=white):
                    has_own_pawn = True
                    break

            if not has_own_pawn:
                files_without_own_pawn += 1

        if total_files == 0:
            return 0.0

        return files_without_own_pawn / total_files

    def _minor_development_score(self, board: Board, *, white: bool) -> float:
        if white:
            home_squares = ["b1", "c1", "f1", "g1"]
        else:
            home_squares = ["b8", "c8", "f8", "g8"]

        undeveloped = 0

        for square in home_squares:
            piece = self._piece_at(board, square)

            if piece == p.NONE:
                continue

            if p.is_white(piece) != white:
                continue

            if p.piece_type(piece) in (p.KNIGHT, p.BISHOP):
                undeveloped += 1

        return (4 - undeveloped) / 4.0

    def _queen_developed(self, board: Board, *, white: bool) -> bool:
        """
        True only if this side still has a queen and it is not on its home square.
        A traded queen is not counted as developed.
        """
        queen_exists = False

        for sq, piece in enumerate(board.get_board()):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) != white:
                continue

            if p.piece_type(piece) != p.QUEEN:
                continue

            queen_exists = True

            home = "d1" if white else "d8"
            return sq != self._idx(board, home)

        return False

    def _pawn_advancement_score(self, board: Board, *, white: bool) -> float:
        score = 0.0

        for sq, piece in enumerate(board.get_board()):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) != white:
                continue

            if p.piece_type(piece) != p.PAWN:
                continue

            _file, rank = self._coord(board, sq)

            if white:
                score += max(0.0, min(1.0, (rank - 2) / 6.0))
            else:
                score += max(0.0, min(1.0, (7 - rank) / 6.0))

        return score / 8.0

    def _passed_pawn_score(self, board: Board, *, white: bool) -> float:
        arr = board.get_board()
        passed = 0

        for sq, piece in enumerate(arr):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) != white:
                continue

            if p.piece_type(piece) != p.PAWN:
                continue

            file_idx, rank = self._coord(board, sq)

            if self._is_passed_pawn(board, file_idx, rank, white=white):
                passed += 1

        return passed / 8.0

    def _is_passed_pawn(self, board: Board, file_idx: int, rank: int, *, white: bool) -> bool:
        enemy_white = not white

        for f in (file_idx - 1, file_idx, file_idx + 1):
            if f < 0 or f > 7:
                continue

            if white:
                ranks = range(rank + 1, 9)
            else:
                ranks = range(rank - 1, 0, -1)

            for r in ranks:
                sq = self._square_from_file_rank(board, f, r)
                if sq is None:
                    continue

                piece = int(board.get_board()[sq])

                if piece == p.NONE:
                    continue

                if p.is_white(piece) == enemy_white and p.piece_type(piece) == p.PAWN:
                    return False

        return True

    def _promotion_pressure_score(self, board: Board, *, white: bool) -> float:
        score = 0.0

        for sq, piece in enumerate(board.get_board()):
            piece = int(piece)

            if piece == p.NONE:
                continue

            if p.is_white(piece) != white:
                continue

            if p.piece_type(piece) != p.PAWN:
                continue

            _file, rank = self._coord(board, sq)

            if white:
                # Only advanced pawns contribute strongly.
                score += max(0.0, rank - 4) / 4.0
            else:
                score += max(0.0, 5 - rank) / 4.0

        return score / 8.0