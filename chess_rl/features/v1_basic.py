# v1_basic.py
from __future__ import annotations

from collections import OrderedDict
import numpy as np
from chess_core.board import Board
from chess_core.move import Move
from chess_core.piece import Piece as p

from .base import FeatureSpec


def _move_key(m: Move) -> int:
    """
    Pack a Move into a single int for hashing/caching.
    Layout (safe, roomy):
      bits  0.. 5  src (0..63)
      bits  6..11  dst (0..63)
      bits 12..19  flags (0..255)
      bits 20..23  promotion (0..15)
    """
    return (
        (m.src_square & 0x3F)
        | ((m.dst_square & 0x3F) << 6)
        | ((m.flags & 0xFF) << 12)
        | ((m.promotion & 0x0F) << 20)
    )


class V1BasicFeatures:
    # Keep this immutable once you start training/checkpointing.
    spec = FeatureSpec(name="features", version="v1_basic", dim=16)

    def __init__(self, cache_size: int = 250_000, dtype=np.float64) -> None:
        # LRU cache: (zkey_before_move, move_key) -> phi(afterstate)
        self._cache_size = int(cache_size)
        self._cache: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()
        self._dtype = np.dtype(dtype)

    def clear_cache(self) -> None:
        self._cache.clear()

    def phi_sa(self, board: Board, move: Move) -> np.ndarray:
        # Fast path: cache hit WITHOUT applying the move
        key = (board._zkey, _move_key(move))
        cache = self._cache
        phi = cache.get(key)
        if phi is not None:
            cache.move_to_end(key)
            return phi

        # Cache miss: compute once using do/undo
        u = board._do_move(move)
        try:
            phi = self._phi_afterstate(board)
            if phi.dtype != self._dtype:
                phi = phi.astype(self._dtype, copy=False)
        finally:
            board._undo_move(u)

        cache[key] = phi
        cache.move_to_end(key)
        if len(cache) > self._cache_size:
            cache.popitem(last=False)  # evict LRU
        return phi

    def _phi_afterstate(self, board: Board) -> np.ndarray:
        # Evaluate in the position AFTER move (opponent to move).

        wp, wn, wb, wr, wq, bp, bn, bb, br, bq = board._mat

        cr = board._castling_rights
        wk  = 1.0 if (cr & Board.WHITE_CASTLE_KINGSIDE) else 0.0
        wq_ = 1.0 if (cr & Board.WHITE_CASTLE_QUEENSIDE) else 0.0
        bk  = 1.0 if (cr & Board.BLACK_CASTLE_KINGSIDE) else 0.0
        bq_ = 1.0 if (cr & Board.BLACK_CASTLE_QUEENSIDE) else 0.0

        stm = 1.0 if board._is_white_to_move else 0.0

        mat = (100*(wp-bp) + 320*(wn-bn) + 330*(wb-bb) + 500*(wr-br) + 900*(wq-bq)) / 1000.0

        # Check features only test the side to move
        if board._is_white_to_move:
            w_in_check = 1.0 if board.is_square_attacked(board._white_king_sq, by_white=False) else 0.0
            b_in_check = 0.0
        else:
            b_in_check = 1.0 if board.is_square_attacked(board._black_king_sq, by_white=True) else 0.0
            w_in_check = 0.0

        phi = np.empty(16, dtype=self._dtype)
        phi[0]  = 1.0
        phi[1]  = mat
        phi[2]  = (wp-bp)/8.0
        phi[3]  = (wn-bn)/2.0
        phi[4]  = (wb-bb)/2.0
        phi[5]  = (wr-br)/2.0
        phi[6]  = (wq-bq)/1.0
        phi[7]  = wk
        phi[8]  = wq_
        phi[9]  = bk
        phi[10] = bq_
        phi[11] = stm
        phi[12] = w_in_check
        phi[13] = b_in_check
        phi[14] = 0.0
        phi[15] = 0.0
        return phi

    def phi_sa_after_move(self, pre_zkey: int, move: Move, board_after: Board) -> np.ndarray:
        """
        Compute phi for (state, move) when the board is ALREADY in the afterstate.
        pre_zkey is board_before_move._zkey (so caching matches phi_sa()).
        """
        key = (pre_zkey, _move_key(move))
        cache = self._cache
        phi = cache.get(key)
        if phi is not None:
            cache.move_to_end(key)
            return phi

        phi = self._phi_afterstate(board_after)
        if phi.dtype != self._dtype:
            phi = phi.astype(self._dtype, copy=False)

        cache[key] = phi
        cache.move_to_end(key)
        if len(cache) > self._cache_size:
            cache.popitem(last=False)
        return phi
