from __future__ import annotations
import numpy as np
from chess_core.board import Board
from chess_core.move import Move
from chess_core.piece import Piece as p
from .base import FeatureSpec

class V1BasicFeatures:
    # Keep this immutable once you start training/checkpointing.
    spec = FeatureSpec(name="features", version="v1_basic", dim=16)

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        u = board._do_move(move)
        try:
            return self._phi_afterstate(board)
        finally:
            board._undo_move(u)

    def _phi_afterstate(self, board: Board) -> np.ndarray:
        # NOTE: this is evaluated in the position AFTER move (opponent to move).
        # Define everything from White’s perspective; policy will max for white, min for black.
        b = board.get_board()
        # material counts
        wp = wn = wb = wr = wq = 0
        bp = bn = bb = br = bq = 0

        for pc in b:
            pc = int(pc)
            if pc == p.NONE: 
                continue
            t = p.piece_type(pc)
            if p.is_white(pc):
                if t == p.PAWN: wp += 1
                elif t == p.KNIGHT: wn += 1
                elif t == p.BISHOP: wb += 1
                elif t == p.ROOK: wr += 1
                elif t == p.QUEEN: wq += 1
            else:
                if t == p.PAWN: bp += 1
                elif t == p.KNIGHT: bn += 1
                elif t == p.BISHOP: bb += 1
                elif t == p.ROOK: br += 1
                elif t == p.QUEEN: bq += 1

        # castling rights (access internal mask; or add getters if you prefer)
        cr = board._castling_rights
        wk = 1.0 if (cr & Board.WHITE_CASTLE_KINGSIDE) else 0.0
        wq_ = 1.0 if (cr & Board.WHITE_CASTLE_QUEENSIDE) else 0.0
        bk = 1.0 if (cr & Board.BLACK_CASTLE_KINGSIDE) else 0.0
        bq_ = 1.0 if (cr & Board.BLACK_CASTLE_QUEENSIDE) else 0.0

        stm = 1.0 if board.get_is_white_to_move() else 0.0  # “white to move” flag

        # simple material score (centipawn-ish, from White view)
        mat = (100*(wp-bp) + 320*(wn-bn) + 330*(wb-bb) + 500*(wr-br) + 900*(wq-bq)) / 1000.0

        # tiny king-in-check indicator (from White view)
        w_in_check = 1.0 if board.in_check(True) else 0.0
        b_in_check = 1.0 if board.in_check(False) else 0.0

        # bias
        phi = np.array([
            1.0,
            mat,
            (wp-bp)/8.0, (wn-bn)/2.0, (wb-bb)/2.0, (wr-br)/2.0, (wq-bq)/1.0,
            wk, wq_, bk, bq_,
            stm,
            w_in_check, b_in_check,
            0.0, 0.0,  # reserved slots for v1 tweaks without breaking dim
        ], dtype=np.float64)

        return phi
