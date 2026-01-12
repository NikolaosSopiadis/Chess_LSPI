from __future__ import annotations
from dataclasses import dataclass
from chess_core.board import Board
from chess_core.piece import Piece as p

@dataclass(frozen=True)
class RewardSpec:
    name: str = "reward"
    version: str = "v1_terminal_plus_potential"

def material_potential(board: Board) -> float:
    b = board.get_board()
    val = 0
    for pc in b:
        pc = int(pc)
        if pc == p.NONE: 
            continue
        t = p.piece_type(pc)
        sgn = 1 if p.is_white(pc) else -1
        if t == p.PAWN: val += 100 * sgn
        elif t == p.KNIGHT: val += 320 * sgn
        elif t == p.BISHOP: val += 330 * sgn
        elif t == p.ROOK: val += 500 * sgn
        elif t == p.QUEEN: val += 900 * sgn
    return val / 1000.0

def step_reward(board_before: Board, board_after: Board, *, alpha: float = 0.05) -> float:
    done, reason = board_after.game_end_state()

    r_terminal = 0.0
    if done:
        # Determine winner by side-to-move having no moves:
        # If checkmate and it's X to move, X lost.
        if reason == "checkmate":
            # side to move is checkmated => loses
            loser_is_white = board_after.get_is_white_to_move()
            r_terminal = -1.0 if loser_is_white else +1.0
        else:
            r_terminal = 0.0

    # potential shaping
    phi0 = material_potential(board_before)
    phi1 = material_potential(board_after)
    return r_terminal + alpha * (phi1 - phi0)
