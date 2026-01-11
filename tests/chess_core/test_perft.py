import pytest
from chess_core.board import Board

def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    moves = board.get_all_legal_moves()
    for m in moves:
        u = board._do_move(m)          # ok for tests
        nodes += perft(board, depth-1)
        board._undo_move(u)
    return nodes

@pytest.mark.parametrize("depth,expected", [
    (1, 20),
    (2, 400),
    (3, 8902),
    (4, 197281),
    (5, 4865609),
])
def test_perft_startpos(depth, expected):
    b = Board()
    assert perft(b, depth) == expected
