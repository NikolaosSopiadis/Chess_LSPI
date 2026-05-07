from __future__ import annotations

import random
from dataclasses import dataclass

from chess_core.board import Board
from chess_core.move import F_CAPTURE, Move
from chess_core.piece import Piece as p

from chess_rl.agents.base import Agent


MATE_SCORE = 1_000_000.0


PIECE_VALUE_CP = {
    p.PAWN: 100.0,
    p.KNIGHT: 320.0,
    p.BISHOP: 330.0,
    p.ROOK: 500.0,
    p.QUEEN: 900.0,
    p.KING: 0.0,
}


@dataclass(slots=True)
class MaterialMinimaxAgent(Agent):
    depth: int = 2
    seed: int = 1

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def pick_move(self, board: Board) -> Move:
        moves = board.get_all_legal_moves()
        if not moves:
            raise RuntimeError("No legal moves available")

        self._order_moves_in_place(board, moves)

        best_score = float("-inf")
        best_moves: list[Move] = []

        alpha = float("-inf")
        beta = float("inf")

        for move in moves:
            with board.temporary_move(move):
                score = -self._negamax(
                    board,
                    depth=self.depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    ply_from_root=1,
                )

            if score > best_score + 1e-12:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) <= 1e-12:
                best_moves.append(move)

            alpha = max(alpha, best_score)

        return self._rng.choice(best_moves)

    def _negamax(
        self,
        board: Board,
        *,
        depth: int,
        alpha: float,
        beta: float,
        ply_from_root: int,
    ) -> float:
        done, reason = board.game_end_state()
        if done:
            return self._terminal_score(board, reason, ply_from_root)

        if depth <= 0:
            return self._eval_side_to_move(board)

        moves = board.get_all_legal_moves()
        if not moves:
            return self._eval_side_to_move(board)

        self._order_moves_in_place(board, moves)

        best = float("-inf")

        for move in moves:
            with board.temporary_move(move):
                score = -self._negamax(
                    board,
                    depth=depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    ply_from_root=ply_from_root + 1,
                )

            if score > best:
                best = score

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best

    def _terminal_score(self, board: Board, reason: str, ply_from_root: int) -> float:
        if reason == "checkmate":
            # Side to move is checkmated, so from side-to-move perspective this is bad.
            return -MATE_SCORE + float(ply_from_root)

        # stalemate, repetition, fifty-move, insufficient material
        return 0.0

    def _eval_side_to_move(self, board: Board) -> float:
        white_score = self._material_white_perspective(board)
        return white_score if board.get_is_white_to_move() else -white_score

    def _material_white_perspective(self, board: Board) -> float:
        score = 0.0

        for pc_raw in board.get_board():
            pc = int(pc_raw)
            if pc == p.NONE:
                continue

            value = PIECE_VALUE_CP.get(p.piece_type(pc), 0.0)

            if p.is_white(pc):
                score += value
            else:
                score -= value

        return score

    def _order_moves_in_place(self, board: Board, moves: list[Move]) -> None:
        def key(move: Move) -> float:
            score = 0.0

            if move.flags & F_CAPTURE:
                captured = int(getattr(move, "captured_piece", p.NONE))
                if captured == p.NONE:
                    captured = int(board.get_board()[move.dst_square])

                moving = int(board.get_board()[move.src_square])

                victim_value = PIECE_VALUE_CP.get(p.piece_type(captured), 0.0)
                attacker_value = PIECE_VALUE_CP.get(p.piece_type(moving), 0.0)

                # MVV-LVA style ordering.
                score += 10_000.0 + 10.0 * victim_value - attacker_value

            move_gives_check = getattr(board, "move_gives_check", None)
            if move_gives_check is not None:
                try:
                    if move_gives_check(move):
                        score += 1_000.0
                except Exception:
                    pass

            return score

        moves.sort(key=key, reverse=True)