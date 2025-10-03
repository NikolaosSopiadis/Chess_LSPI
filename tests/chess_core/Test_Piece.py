import pytest
from chess_core.Piece import Piece

@pytest.fixture
def p():
    return Piece()

def test_make_piece_and_constants(p):
    wp = p.makePiece(Piece.PAWN, Piece.WHITE)
    bp = p.makePiece(Piece.PAWN, Piece.BLACK)
    assert wp == Piece.WHITE_PAWN
    assert bp == Piece.BLACK_PAWN

def test_type_and_colour(p):
    wp = p.makePiece(Piece.PAWN, Piece.WHITE)
    bp = p.makePiece(Piece.PAWN, Piece.BLACK)
    assert p.pieceType(wp) == Piece.PAWN
    assert p.pieceColour(wp) == Piece.WHITE
    assert p.pieceType(bp) == Piece.PAWN
    assert p.pieceColour(bp) == Piece.BLACK

def test_colour_helpers(p):
    wp = p.makePiece(Piece.PAWN, Piece.WHITE)
    bp = p.makePiece(Piece.PAWN, Piece.BLACK)
    assert p.isColour(wp, Piece.WHITE)
    assert not p.isColour(bp, Piece.WHITE)
    assert p.isWhite(wp)
    assert not p.isWhite(bp)

def test_masks_sanity(p):
    wp = p.makePiece(Piece.PAWN, Piece.WHITE)
    assert (wp & Piece.TYPE_MASK) == Piece.PAWN
    assert (wp & Piece.COLOUR_MASK) == Piece.WHITE

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
    piece = p.makePiece(kind, Piece.WHITE)
    assert p.isOrthogonalSlider(piece) is is_orth
    assert p.isDiagonalSlider(piece)  is is_diag
    assert p.isSlidingPiece(piece)    is is_slide

def test_none_piece_behavior(p):
    assert p.pieceType(Piece.NONE) == 0
    assert p.pieceColour(Piece.NONE) == 0
    assert not p.isColour(Piece.NONE, Piece.WHITE)
    assert not p.isWhite(Piece.NONE)
