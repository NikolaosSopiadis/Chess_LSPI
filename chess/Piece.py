

class Piece:
    # Piece Types
    NONE: int   = 0
    PAWN: int   = 1
    KNIGHT: int = 2
    BISHOP: int = 3
    ROOK: int   = 4
    QUEEN: int  = 5
    KING: int   = 6
 
    # Piece Colours
    WHITE: int = 0
    BLACK: int = 8

    # Pieces
    WHITE_PAWN: int   = PAWN   | WHITE # 1
    WHITE_KNIGHT: int = KNIGHT | WHITE # 2
    WHITE_BISHOP: int = BISHOP | WHITE # 3
    WHITE_ROOK: int   = ROOK   | WHITE # 4
    WHITE_QUEEN: int  = QUEEN  | WHITE # 5
    WHITE_KING: int   = KING   | WHITE # 6

    BLACK_PAWN: int   = PAWN   | BLACK # 9
    BLACK_KNIGHT: int = KNIGHT | BLACK # 10
    BLACK_BISHOP: int = BISHOP | BLACK # 11
    BLACK_ROOK: int   = ROOK   | BLACK # 12
    BLACK_QUEEN: int  = QUEEN  | BLACK # 13
    BLACK_KING: int   = KING   | BLACK # 14
    
    # Bit Masks
    TYPE_MASK: int  = 0b0111
    COLOUR_MASK:int = 0b1000

    def makePiece(self, pieceType: int, pieceColour: int) -> int:
        return pieceType | pieceColour

    def isColour(self, piece: int, colour: int):
        return (piece & self.COLOUR_MASK) == colour and piece != 0

    def isWhite(self, piece: int):
        return self.isColour(piece, self.WHITE)

    def pieceColour(self, piece: int):
        return piece & self.COLOUR_MASK

    def pieceType(self, piece: int):
        return piece & self.TYPE_MASK

    def isOrthogonalSlider(self, piece: int):
        return self.pieceType(piece) == self.ROOK or self.pieceType(piece) == self.QUEEN

    def isDiagonalSlider(self, piece: int):
        return self.pieceType(piece) == self.BISHOP or self.pieceType(piece) == self.QUEEN

    def isSlidingPiece(self, piece: int):
        return self.isOrthogonalSlider(piece) or self.isDiagonalSlider(piece)
    

