from chess.Piece import Piece

def run_tests():
    p = Piece()

    # makePiece and basics
    wp = p.makePiece(Piece.PAWN, Piece.WHITE)
    assert wp == Piece.WHITE_PAWN, f"wp expected {Piece.WHITE_PAWN}, got {wp}"

    bp = p.makePiece(Piece.PAWN, Piece.BLACK)
    assert bp == Piece.BLACK_PAWN, f"bp expected {Piece.BLACK_PAWN}, got {bp}"

    # type and colour extraction
    assert p.pieceType(wp) == Piece.PAWN
    assert p.pieceColour(wp) == Piece.WHITE

    assert p.pieceType(bp) == Piece.PAWN
    assert p.pieceColour(bp) == Piece.BLACK

    # isColour / isWhite
    assert p.isColour(wp, Piece.WHITE) is True
    assert p.isColour(bp, Piece.WHITE) is False
    assert p.isWhite(wp) is True
    assert p.isWhite(bp) is False

    # TYPE_MASK / COLOUR_MASK behavior (sanity)
    assert (wp & Piece.TYPE_MASK) == Piece.PAWN
    assert (wp & Piece.COLOUR_MASK) == Piece.WHITE

    # slider checks
    wq = p.makePiece(Piece.QUEEN, Piece.WHITE)
    wr = p.makePiece(Piece.ROOK, Piece.WHITE)
    wb = p.makePiece(Piece.BISHOP, Piece.WHITE)
    wn = p.makePiece(Piece.KNIGHT, Piece.WHITE)
    wk = p.makePiece(Piece.KING, Piece.WHITE)

    assert p.isOrthogonalSlider(wr) is True
    assert p.isOrthogonalSlider(wq) is True
    assert p.isOrthogonalSlider(wb) is False

    assert p.isDiagonalSlider(wb) is True
    assert p.isDiagonalSlider(wq) is True
    assert p.isDiagonalSlider(wr) is False

    assert p.isSlidingPiece(wq) is True
    assert p.isSlidingPiece(wr) is True
    assert p.isSlidingPiece(wb) is True
    assert p.isSlidingPiece(wn) is False
    assert p.isSlidingPiece(wk) is False

    # NONE / invalid checks
    assert p.pieceType(Piece.NONE) == 0
    assert p.pieceColour(Piece.NONE) == 0
    assert p.isColour(Piece.NONE, Piece.WHITE) is False
    assert p.isWhite(Piece.NONE) is False

    print("All quick tests passed!")

# Run quick tests
if __name__ == "__main__":
    run_tests()
