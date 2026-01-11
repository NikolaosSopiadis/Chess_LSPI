import numpy as np
from chess_core.board import Board

def test_do_undo_restores_board_state():
    b = Board()
    before = b.get_board().copy()
    moves = b.get_all_legal_moves()
    for m in moves[:10]:   # sample
        u = b._do_move(m)
        b._undo_move(u)
        assert np.array_equal(b.get_board(), before)
