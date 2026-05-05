from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent, AgentInfo
from chess_rl.features.registry import get as get_features
# from chess_rl.policy.greedy import greedy_move
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


@dataclass
class LSPITacticalAgent(Agent):
    info: AgentInfo
    w: np.ndarray
    feature_name: str = "v1_basic"  # registry key

    # def pick_move(self, board: Board) -> Move:
    #     feats = get_features(self.feature_name)
    #     return greedy_move(board, self.w, feats)
    def pick_move(self, board: Board) -> Move:
        feats = get_features(self.feature_name)
        moves = board.get_all_legal_moves()

        if not moves:
            raise ValueError("No legal moves")

        white_to_move = board.get_is_white_to_move()

        base_material = float(material_potential(board))
        before_mover_advantage = base_material if white_to_move else -base_material

        best_move: Move | None = None
        best_score: float = 0.0

        for move in moves:
            # Do the move once, compute phi and all policy-time score adjustments
            # in the same afterstate.
            with board.temporary_move(move):
                phi = feats.phi_afterstate(board)
                score = float(self.w @ phi)

                score = self._adjust_score_for_draw_risk_after_move(
                    board,
                    score,
                    mover_was_white=white_to_move,
                )

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

    def save(self, path: str) -> None:
        path = str(path)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.info.name,
            "version": self.info.version,
            "feature_name": self.feature_name,
        }

        np.savez_compressed(path, w=self.w.astype(np.float64), meta=json.dumps(meta))

    @classmethod
    def load(cls, path: str) -> "LSPITacticalAgent":
        data = np.load(path, allow_pickle=False)
        w = data["w"]
        meta = json.loads(str(data["meta"]))
        return cls(
            info=AgentInfo(name=meta["name"], version=meta["version"]),
            w=w,
            feature_name=meta.get("feature_name", "v1_basic"),
        )

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
        AHEAD_THRESHOLD = 0.30

        done, reason = board.game_end_state()
        material = float(material_potential(board))

        mover_advantage = material if mover_was_white else -material

        if mover_advantage <= AHEAD_THRESHOLD:
            return score

        # Drawing while ahead is bad.
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

        # Approaching repetition while ahead is bad.
        rep_count = board.current_repetition_count()
        if rep_count >= 2:
            if mover_was_white:
                score -= REPEAT_PENALTY
            else:
                score += REPEAT_PENALTY

        # Approaching fifty-move draw while ahead is bad.
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

        This avoids obvious one-move queen/rook/bishop hangs while not heavily
        penalizing equal trades. Example:

        Before move: material = equal
        Candidate: Qxd8+
        Opponent: Kxd8

        Final material is still equal, so this should not be treated as a queen blunder.
        """
        LOSS_THRESHOLD = 0.10   # one pawn
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

        # White maximizes score. Bad white moves should be scored lower.
        # Black minimizes score. Bad black moves should be scored higher.
        if mover_was_white:
            score -= penalty
        else:
            score += penalty

        return score