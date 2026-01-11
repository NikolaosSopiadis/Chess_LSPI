from chess_core.board import Board

def snapshot(b: Board):
    # Use bytes() to make it immutable and cheap-to-compare
    return (
        bytes(b.get_board()),
        b._is_white_to_move,
        b._castling_rights,
        b._en_passant_target,
        b._halfmove_clock,
        b._white_king_sq,
        b._black_king_sq,
    )

def test_do_undo_restores_board_state():
    b = Board()
    before = snapshot(b)

    moves = b.get_all_legal_moves()
    for m in moves[:10]:
        u = b._do_move(m)
        b._undo_move(u)
        assert snapshot(b) == before
