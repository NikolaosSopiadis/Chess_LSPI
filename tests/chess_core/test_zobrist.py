import random

import pytest

from chess_core.board import Board
from chess_core.move import MoveFlag, Promotion


# ----------------------------
# Helpers / fixtures
# ----------------------------

@pytest.fixture
def board() -> Board:
    return Board()


def assert_state_consistent(b: Board) -> None:
    assert b._zkey == b._recompute_zobrist()


def move_by_squares(b: Board, src: int, dst: int):
    """Find a legal move by (src,dst). Fail with a useful message if not found."""
    for m in b.get_all_legal_moves():
        if m.src_square == src and m.dst_square == dst:
            return m
    pytest.fail(f"No legal move found: {b.idx_to_algebraic(src)} -> {b.idx_to_algebraic(dst)}")


def do_and_assert(b: Board, move):
    """Do move and verify incremental zobrist stays correct."""
    u = b._do_move(move)
    assert_state_consistent(b)
    return u


def undo_and_assert(b: Board, undo):
    b._undo_move(undo)
    assert_state_consistent(b)


# ----------------------------
# Tests
# ----------------------------

def test_zobrist_matches_recompute_random_play_and_undo(board: Board):
    rng = random.Random(1234)

    b = board
    start_board = bytes(b.get_board())
    start_key = b._zkey

    undos = []
    for _ in range(200):
        moves = b.get_all_legal_moves()
        assert moves  # usually won't end in 200 random plies
        undos.append(do_and_assert(b, rng.choice(moves)))

    while undos:
        undo_and_assert(b, undos.pop())

    assert bytes(b.get_board()) == start_board
    assert b._zkey == start_key
    assert b._rep_stack == [start_key]
    assert b._rep_counts == {start_key: 1}


@pytest.mark.parametrize(
    "fen, flag",
    [
        # White pawn on e5, black pawn just moved d7->d5; ep square d6, so exd6 ep exists
        ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", MoveFlag.EN_PASSANT),
        # Simplified position where castling exists
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", MoveFlag.CASTLE),
    ],
)
def test_zobrist_special_move_roundtrip(board: Board, fen: str, flag: MoveFlag):
    b = board
    b.init_board(fen)

    moves = [m for m in b.get_all_legal_moves() if m.check_flag(flag)]
    assert moves, f"Expected at least one move with flag {flag}"

    u = do_and_assert(b, moves[0])
    undo_and_assert(b, u)


def test_zobrist_promotion(board: Board):
    b = board
    b.init_board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    promo_moves = [m for m in b.get_all_legal_moves() if m.promotion != Promotion.NONE]
    assert promo_moves

    u = do_and_assert(b, promo_moves[0])
    undo_and_assert(b, u)


def test_threefold_repetition_knight_shuffle(board: Board):
    b = board
    b.init_board("4k3/8/8/8/8/8/5N2/4K3 w - - 0 1")

    f2 = b.algebraic_to_idx("f2")
    g4 = b.algebraic_to_idx("g4")
    e8 = b.algebraic_to_idx("e8")
    e7 = b.algebraic_to_idx("e7")

    def cycle_once():
        do_and_assert(b, move_by_squares(b, f2, g4))  # white
        do_and_assert(b, move_by_squares(b, e8, e7))  # black
        do_and_assert(b, move_by_squares(b, g4, f2))  # white
        do_and_assert(b, move_by_squares(b, e7, e8))  # black

    # Initial position is already counted once in init_board.
    cycle_once()
    assert b._rep_counts.get(b._rep_stack[0], 0) >= 2

    cycle_once()
    assert b.is_threefold_repetition()
