import pytest
from chess_core.piece import Piece

@pytest.fixture
def p():
    return Piece()

def test_make_piece_and_constants(p):
    wp = p.make_piece(Piece.PAWN, Piece.WHITE)
    bp = p.make_piece(Piece.PAWN, Piece.BLACK)
    assert wp == Piece.WHITE_PAWN
    assert bp == Piece.BLACK_PAWN

def test_type_and_color(p):
    wp = p.make_piece(Piece.PAWN, Piece.WHITE)
    bp = p.make_piece(Piece.PAWN, Piece.BLACK)
    assert p.piece_type(wp) == Piece.PAWN
    assert p.piece_color(wp) == Piece.WHITE
    assert p.piece_type(bp) == Piece.PAWN
    assert p.piece_color(bp) == Piece.BLACK

def test_color_helpers(p):
    wp = p.make_piece(Piece.PAWN, Piece.WHITE)
    bp = p.make_piece(Piece.PAWN, Piece.BLACK)
    assert p.is_color(wp, Piece.WHITE)
    assert not p.is_color(bp, Piece.WHITE)
    assert p.is_white(wp)
    assert not p.is_white(bp)

def test_masks_sanity(p):
    wp = p.make_piece(Piece.PAWN, Piece.WHITE)
    assert (wp & Piece.TYPE_MASK) == Piece.PAWN
    assert (wp & Piece.COLOR_MASK) == Piece.WHITE

@pytest.mark.parametrize(
    "kind,is_orth,is_diag,is_slide",
    [
        (Piece.ROOK,   True,  False, True),
        (Piece.BISHOP, False, True,  True),
        (Piece.QUEEN,  True,  True,  True),
        (Piece.KNIGHT, False, False, False),
        (Piece.KING,   False, False, False),
        (Piece.PAWN,   False, False, False),
    ]
)
def test_slider_checks(p, kind, is_orth, is_diag, is_slide):
    piece = p.make_piece(kind, Piece.WHITE)
    assert p.is_orthogonal_slider(piece) is is_orth
    assert p.is_diagonal_slider(piece)  is is_diag
    assert p.is_sliding_piece(piece)    is is_slide

def test_none_piece_behavior(p):
    assert p.piece_type(Piece.NONE) == 0
    assert p.piece_color(Piece.NONE) == 0
    assert not p.is_color(Piece.NONE, Piece.WHITE)
    assert not p.is_white(Piece.NONE)
