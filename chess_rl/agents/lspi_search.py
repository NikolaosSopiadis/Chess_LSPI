from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from chess_core.board import Board
from chess_core.move import Move
from chess_core.piece import Piece as p
from chess_rl.agents.base import Agent, AgentInfo
from chess_rl.features.registry import get as get_features
from chess_rl.features.base import FeatureExtractor
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


MATE_SCORE = 100.0
DRAW_SCORE = 0.0


@dataclass
class LSPISearchAgent(Agent):
    """
    LSPI checkpoint + shallow minimax search.

    This keeps LSPIV1Agent archived as the fast/safe 1-ply version.
    This class is for the next experimental agent.
    """

    info: AgentInfo
    w: np.ndarray
    feature_name: str = "v3_basic"

    # Search settings.
    depth: int = 2

    # None means full-width search.
    # For depth=2, None is usually okay.
    # For depth=3, use something like 8-16.
    max_branch: Optional[int] = None

    # Keep the successful root-level safety patches.
    use_draw_safety: bool = True
    use_tactical_safety: bool = True

    def pick_move(self, board: Board) -> Move:
        feats = get_features(self.feature_name)
        moves = board.get_all_legal_moves()

        if not moves:
            raise ValueError("No legal moves")

        white_to_move = board.get_is_white_to_move()
        before_material = float(material_potential(board))
        before_mover_advantage = before_material if white_to_move else -before_material

        best_move: Move | None = None
        best_score: float = -math.inf if white_to_move else math.inf

        # Search all root moves. Do not branch-limit root by default.
        ordered_root_moves = self._order_moves(board, moves, feats)

        for move in ordered_root_moves:
            with board.temporary_move(move):
                score = self._search(
                    board,
                    feats,
                    depth_left=max(0, self.depth - 1),
                    alpha=-math.inf,
                    beta=math.inf,
                )

                # Keep the already successful policy-time wrappers at the root.
                if self.use_draw_safety:
                    score = self._adjust_score_for_draw_risk_after_move(
                        board,
                        score,
                        mover_was_white=white_to_move,
                    )

                if self.use_tactical_safety:
                    score = self._adjust_score_for_tactical_safety_after_move(
                        board,
                        score,
                        mover_was_white=white_to_move,
                        before_mover_advantage=before_mover_advantage,
                    )

            if best_move is None:
                best_move = move
                best_score = score
                continue

            if white_to_move:
                if score > best_score:
                    best_move = move
                    best_score = score
            else:
                if score < best_score:
                    best_move = move
                    best_score = score

        assert best_move is not None
        return best_move

    def _search(
        self,
        board: Board,
        feats: FeatureExtractor,
        *,
        depth_left: int,
        alpha: float,
        beta: float,
    ) -> float:
        terminal = self._terminal_score(board)
        if terminal is not None:
            return terminal

        if depth_left <= 0:
            return self._evaluate(board, feats)

        moves = board.get_all_legal_moves()
        if not moves:
            # Should normally be caught by game_end_state(), but keep it safe.
            return self._evaluate(board, feats)

        white_to_move = board.get_is_white_to_move()

        ordered_moves = self._order_moves(board, moves, feats)

        if self.max_branch is not None and self.max_branch > 0:
            ordered_moves = ordered_moves[: self.max_branch]

        if white_to_move:
            value = -math.inf

            for move in ordered_moves:
                with board.temporary_move(move):
                    child = self._search(
                        board,
                        feats,
                        depth_left=depth_left - 1,
                        alpha=alpha,
                        beta=beta,
                    )

                value = max(value, child)
                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        else:
            value = math.inf

            for move in ordered_moves:
                with board.temporary_move(move):
                    child = self._search(
                        board,
                        feats,
                        depth_left=depth_left - 1,
                        alpha=alpha,
                        beta=beta,
                    )

                value = min(value, child)
                beta = min(beta, value)

                if alpha >= beta:
                    break

            return value

    def _evaluate(self, board: Board, feats: FeatureExtractor) -> float:
        terminal = self._terminal_score(board)
        if terminal is not None:
            return terminal

        phi = feats.phi_afterstate(board)
        return float(self.w @ phi)

    def _terminal_score(self, board: Board) -> float | None:
        done, reason = board.game_end_state()

        if not done:
            return None

        if reason == "checkmate":
            # Side to move is checkmated.
            # White-perspective score:
            #   black to move and checkmated => white wins => positive
            #   white to move and checkmated => black wins => negative
            return -MATE_SCORE if board.get_is_white_to_move() else MATE_SCORE

        return DRAW_SCORE

    def _order_moves(
        self,
        board: Board,
        moves: list[Move],
        feats: FeatureExtractor,
    ) -> list[Move]:
        """
        Cheap move ordering.

        Avoid calling phi_afterstate() here. v3 features are expensive, and
        evaluating every move just for ordering duplicates work.

        Order:
        captures/promotions first,
        quiet moves later.

        This is not perfect, but it is much faster.
        """
        arr = board.get_board()

        scored: list[tuple[float, Move]] = []

        for move in moves:
            captured = arr[move.dst_square]
            capture_score = self._piece_value(captured)

            promo_score = 0.0
            if move.promotion:
                promo_score = self._promotion_value(move.promotion)

            # MVV-like ordering. We mostly care that forcing/material moves are first.
            order_score = capture_score + promo_score

            scored.append((order_score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _score, m in scored]

    def _adjust_score_for_draw_risk_after_move(
        self,
        board: Board,
        score: float,
        *,
        mover_was_white: bool,
    ) -> float:
        DRAW_PENALTY = 0.75
        REPEAT_PENALTY = 0.35
        FIFTY_MOVE_PENALTY = 0.35

        # material_potential units:
        # pawn = 0.10, rook = 0.50, queen = 0.90
        AHEAD_THRESHOLD = 0.30

        done, reason = board.game_end_state()
        material = float(material_potential(board))

        mover_advantage = material if mover_was_white else -material

        if mover_advantage <= AHEAD_THRESHOLD:
            return score

        if done and reason in {
            "stalemate",
            "threefold repetition",
            "fifty-move rule",
            "insufficient material",
        }:
            if mover_was_white:
                score -= DRAW_PENALTY
            else:
                score += DRAW_PENALTY

        rep_count = board.current_repetition_count()
        if rep_count >= 2:
            if mover_was_white:
                score -= REPEAT_PENALTY
            else:
                score += REPEAT_PENALTY

        halfmove = getattr(board, "_halfmove_clock", 0)
        if halfmove >= 80:
            pressure = (halfmove - 80) / 20.0
            pressure = max(0.0, min(1.0, pressure))
            penalty = FIFTY_MOVE_PENALTY * pressure

            if mover_was_white:
                score -= penalty
            else:
                score += penalty

        return score

    def _adjust_score_for_tactical_safety_after_move(
        self,
        board: Board,
        score: float,
        *,
        mover_was_white: bool,
        before_mover_advantage: float,
    ) -> float:
        """
        Penalize moves that allow the opponent to immediately leave us materially worse
        than we were before making the candidate move.
        """
        LOSS_THRESHOLD = 0.10
        PENALTY_SCALE = 1.25
        PENALTY_CAP = 2.50

        done, _reason = board.game_end_state()
        if done:
            return score

        replies = board.get_all_legal_moves()
        if not replies:
            return score

        worst_after_reply_advantage = before_mover_advantage

        for reply in replies:
            with board.temporary_move(reply):
                material = float(material_potential(board))
                adv = material if mover_was_white else -material

            if adv < worst_after_reply_advantage:
                worst_after_reply_advantage = adv

        material_loss = before_mover_advantage - worst_after_reply_advantage

        if material_loss <= LOSS_THRESHOLD:
            return score

        penalty = PENALTY_SCALE * (material_loss - LOSS_THRESHOLD)
        penalty = min(PENALTY_CAP, penalty)

        if mover_was_white:
            score -= penalty
        else:
            score += penalty

        return score

    def save(self, path: str) -> None:
        path = str(path)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.info.name,
            "version": self.info.version,
            "feature_name": self.feature_name,
            "depth": self.depth,
            "max_branch": self.max_branch,
        }

        np.savez_compressed(path, w=self.w.astype(np.float64), meta=json.dumps(meta))

    @classmethod
    def load(
        cls,
        path: str,
        *,
        depth: int = 2,
        max_branch: Optional[int] = None,
        use_draw_safety: bool = True,
        use_tactical_safety: bool = True,
    ) -> "LSPISearchAgent":
        data = np.load(path, allow_pickle=False)
        w = data["w"]
        meta = json.loads(str(data["meta"]))

        return cls(
            info=AgentInfo(name="LSPI search", version="v1"),
            w=w,
            feature_name=meta.get("feature_name", "v3_basic"),
            depth=depth,
            max_branch=max_branch,
            use_draw_safety=use_draw_safety,
            use_tactical_safety=use_tactical_safety,
        )

    def _piece_value(self, piece: int) -> float:
        if piece == p.NONE:
            return 0.0

        t = p.piece_type(piece)

        if t == p.PAWN:
            return 1.0
        if t == p.KNIGHT:
            return 3.2
        if t == p.BISHOP:
            return 3.3
        if t == p.ROOK:
            return 5.0
        if t == p.QUEEN:
            return 9.0
        if t == p.KING:
            return 100.0

        return 0.0


    def _promotion_value(self, promotion: int) -> float:
        if promotion == p.QUEEN:
            return 9.0
        if promotion == p.ROOK:
            return 5.0
        if promotion == p.BISHOP:
            return 3.3
        if promotion == p.KNIGHT:
            return 3.2
        return 0.0