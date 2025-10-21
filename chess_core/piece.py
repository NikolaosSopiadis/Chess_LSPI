from typing import Final

class Piece:
    # Piece Types
    NONE: Final[int]   = 0
    PAWN: Final[int]   = 1
    KNIGHT: Final[int] = 2
    BISHOP: Final[int] = 3
    ROOK: Final[int]   = 4
    QUEEN: Final[int]  = 5
    KING: Final[int]   = 6
 
    # Piece Colors
    WHITE: Final[int] = 0
    BLACK: Final[int] = 8

    # Pieces
    WHITE_PAWN: Final[int]   = PAWN   | WHITE # 1
    WHITE_KNIGHT: Final[int] = KNIGHT | WHITE # 2
    WHITE_BISHOP: Final[int] = BISHOP | WHITE # 3
    WHITE_ROOK: Final[int]   = ROOK   | WHITE # 4
    WHITE_QUEEN: Final[int]  = QUEEN  | WHITE # 5
    WHITE_KING: Final[int]   = KING   | WHITE # 6

    BLACK_PAWN: Final[int]   = PAWN   | BLACK # 9
    BLACK_KNIGHT: Final[int] = KNIGHT | BLACK # 10
    BLACK_BISHOP: Final[int] = BISHOP | BLACK # 11
    BLACK_ROOK: Final[int]   = ROOK   | BLACK # 12
    BLACK_QUEEN: Final[int]  = QUEEN  | BLACK # 13
    BLACK_KING: Final[int]   = KING   | BLACK # 14
    
    # Bit Masks
    TYPE_MASK: Final[int]  = 0b0111
    COLOR_MASK:Final[int] = 0b1000

    @staticmethod
    def make_piece(piece_type: int, piece_color: int) -> int:
        return piece_type | piece_color

    @staticmethod
    def is_color(piece: int, color: int) -> bool:
        return (piece & Piece.COLOR_MASK) == color and piece != 0

    @staticmethod
    def is_white(piece: int) -> bool:
        return Piece.is_color(piece, Piece.WHITE)

    @staticmethod
    def piece_color(piece: int) -> int:
        return piece & Piece.COLOR_MASK

    @staticmethod
    def piece_type(piece: int) -> int:
        return piece & Piece.TYPE_MASK

    @staticmethod
    def is_orthogonal_slider(piece: int) -> bool:
        return Piece.piece_type(piece) == Piece.ROOK or Piece.piece_type(piece) == Piece.QUEEN

    @staticmethod
    def is_diagonal_slider(piece: int) -> bool:
        return Piece.piece_type(piece) == Piece.BISHOP or Piece.piece_type(piece) == Piece.QUEEN

    @staticmethod
    def is_sliding_piece(piece: int) -> bool:
        return Piece.is_orthogonal_slider(piece) or Piece.is_diagonal_slider(piece)
    

