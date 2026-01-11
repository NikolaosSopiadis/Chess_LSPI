import time
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


def bench_perft(board, depth: int) -> int:
    t0 = time.perf_counter()
    n = perft(board, depth)   # your existing perft
    dt = time.perf_counter() - t0
    print(f"depth {depth}: {n} nodes in {dt:.3f}s -> {n/dt:,.0f} nps")
    return n


@pytest.mark.parametrize("depth,expected", [
    (1, 20),
    (2, 400),
    (3, 8902),
    (4, 197281),
    (5, 4865609),
])
def test_perft_pos1(depth, expected):
    b = Board()
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert bench_perft(b, depth) == expected


@pytest.mark.parametrize("depth,expected", [
    (1, 48),
    (2, 2039),
    (3, 97862),
    (4, 4085603),
])
def test_perft_pos2(depth, expected):
    b = Board()
    b.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ")
    assert bench_perft(b, depth) == expected


@pytest.mark.parametrize("depth,expected", [
    (1, 14),
    (2, 191),
    (3, 2812),
    (4, 43238),
    (5, 674624),
    (6, 11030083),
])
def test_perft_pos3(depth, expected):
    b = Board()
    b.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 ")
    assert bench_perft(b, depth) == expected


@pytest.mark.parametrize("depth,expected", [
    (1, 6),
    (2, 264),
    (3, 9467),
    (4, 422333),
])
def test_perft_pos4(depth, expected):
    b = Board()
    b.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    assert bench_perft(b, depth) == expected


@pytest.mark.parametrize("depth,expected", [
    (1, 44),
    (2, 1486),
    (3, 62379),
    (4, 2103487),
])
def test_perft_pos5(depth, expected):
    b = Board()
    b.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ")
    assert bench_perft(b, depth) == expected

    
@pytest.mark.parametrize("depth,expected", [
    (1, 46),
    (2, 2079),
    (3, 89890),
    (4, 3894594),
])
def test_perft_pos6(depth, expected):
    b = Board()
    b.set_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
    assert bench_perft(b, depth) == expected