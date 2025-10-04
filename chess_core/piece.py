

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
    COLOR_MASK:int = 0b1000

    @staticmethod
    def make_piece(piece_type: int, piece_color: int) -> int:
        return piece_type | piece_color

    @staticmethod
    def is_colour(piece: int, colour: int):
        return (piece & Piece.COLOR_MASK) == colour and piece != 0

    @staticmethod
    def is_white(piece: int):
        return Piece.is_colour(piece, Piece.WHITE)

    @staticmethod
    def piece_color(piece: int):
        return piece & Piece.COLOR_MASK

    @staticmethod
    def piece_type(piece: int):
        return piece & Piece.TYPE_MASK

    @staticmethod
    def is_orthogonal_slider(piece: int):
        return Piece.piece_type(piece) == Piece.ROOK or Piece.piece_type(piece) == Piece.QUEEN

    @staticmethod
    def is_diagonal_slider(piece: int):
        return Piece.piece_type(piece) == Piece.BISHOP or Piece.piece_type(piece) == Piece.QUEEN

    @staticmethod
    def is_sliding_piece(piece: int):
        return Piece.is_orthogonal_slider(piece) or Piece.is_diagonal_slider(piece)
    

