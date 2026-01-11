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
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert perft(b, depth) == expected

@pytest.mark.parametrize("depth,expected", [
    (1, 48),
    (2, 2039),
    (3, 97862),
    (4, 4085603),
])
def test_perft_pos2(depth, expected):
    b = Board()
    b.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ")
    assert perft(b, depth) == expected

@pytest.mark.parametrize("depth,expected", [
    (1, 44),
    (2, 1486),
    (3, 62379),
    (4, 2103487),
])
def test_perft_pos5(depth, expected):
    b = Board()
    b.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ")
    assert perft(b, depth) == expected