from __future__ import annotations

from typing import Final, Sequence, Iterator
from dataclasses import dataclass
import random
from contextlib import contextmanager

from chess_core.piece import Piece as p
from chess_core.move import F_CAPTURE, F_CASTLE, F_DOUBLE_PAWN, F_EN_PASSANT, F_PROMOTION, PROMO_BISHOP, PROMO_KNIGHT, PROMO_NONE, PROMO_QUEEN, PROMO_ROOK, Move

DEBUG_MAT = False

# Perforamance TODO:
#      - Stop calling is_square_attacked() twice for castling legality check
#      - BIG: replace Move objects with a packed integer encoding for faster move generation and lookup
#          - 6 bits src, 6 bits dst, 5 bits flags, 3 bits promotion
#      - Reduce undo overhead by making undo a tuple

# Precomputed arrays for piece properties
_ALL_PIECES = [
    p.NONE,
    p.WHITE_PAWN, p.WHITE_KNIGHT, p.WHITE_BISHOP, p.WHITE_ROOK, p.WHITE_QUEEN, p.WHITE_KING,
    p.BLACK_PAWN, p.BLACK_KNIGHT, p.BLACK_BISHOP, p.BLACK_ROOK, p.BLACK_QUEEN, p.BLACK_KING,
]

_MAX_PC = max(_ALL_PIECES)
IS_WHITE = [False] * (_MAX_PC + 1)
PTYPE    = [0]     * (_MAX_PC + 1)  # p.PAWN / p.KNIGHT / ...

for pc in _ALL_PIECES:
    if pc == p.NONE:
        continue
    IS_WHITE[pc] = p.is_white(pc)
    PTYPE[pc]    = p.piece_type(pc)
    
#######################################
### Precompute attack tables + rays ###
#######################################

def _king_neighbors(sq: int) -> list[int]:
    f = sq & 7
    r = sq >> 3
    out = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf = f + df
            nr = r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                out.append(nf | (nr << 3))
    return out

def _knight_targets(sq: int) -> list[int]:
    f = sq & 7
    r = sq >> 3
    out = []
    jumps = ((-1,+2), (+1,+2), (-2,+1), (+2,+1),
             (-2,-1), (+2,-1), (-1,-2), (+1,-2))
    for df, dr in jumps:
        nf = f + df
        nr = r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            out.append(nf | (nr << 3))
    return out


KNIGHT_DELTAS = (17, 15, 10, 6, -6, -10, -15, -17)
KING_DELTAS   = (1, -1, 8, -8, 9, 7, -7, -9)
KING_MOVES   = [_king_neighbors(sq) for sq in range(64)]
KNIGHT_MOVES = [_knight_targets(sq) for sq in range(64)]
# attacker squares FROM which a pawn of a given color attacks sq
PAWN_ATTACKERS_WHITE = [[] for _ in range(64)]  # white pawn attacks upward; attackers are "downward"
PAWN_ATTACKERS_BLACK = [[] for _ in range(64)]  # black pawn attacks downward; attackers are "upward"

for sq in range(64):
    f = sq & 7
    r = sq >> 3
    # white pawn attacks sq from (f-1,r-1) and (f+1,r-1)
    if r - 1 >= 0:
        if f - 1 >= 0: PAWN_ATTACKERS_WHITE[sq].append((f - 1) | ((r - 1) << 3))
        if f + 1 < 8:  PAWN_ATTACKERS_WHITE[sq].append((f + 1) | ((r - 1) << 3))
    # black pawn attacks sq from (f-1,r+1) and (f+1,r+1)
    if r + 1 < 8:
        if f - 1 >= 0: PAWN_ATTACKERS_BLACK[sq].append((f - 1) | ((r + 1) << 3))
        if f + 1 < 8:  PAWN_ATTACKERS_BLACK[sq].append((f + 1) | ((r + 1) << 3))

# board.py (module-level)

DIRS = (1, -1, 8, -8, 9, 7, -7, -9)  # E,W,N,S,NE,NW,SE,SW

def _ray_from(sq: int, step: int) -> list[int]:
    out = []
    f0 = sq & 7
    cur = sq
    while True:
        nxt = cur + step
        if not (0 <= nxt < 64):
            return out
        # prevent wrap on horizontal/diagonal
        f1 = nxt & 7
        if step == 1 and f1 == 0:  # wrapped h->a
            return out
        if step == -1 and f1 == 7: # wrapped a->h
            return out
        # diagonals must change file by 1 each step
        if step in (9, -7) and f1 == 0:  # wrapped
            return out
        if step in (7, -9) and f1 == 7:  # wrapped
            return out

        out.append(nxt)
        cur = nxt

# used tuple instead of list for immutability and better cache friendliness
RAYS: Final[tuple[tuple[tuple[int, ...], ...], ...]] = tuple(
    tuple(tuple(_ray_from(sq, step)) for step in DIRS)
    for sq in range(64)
)

# indices in RAYS
E,W,N,S,NE,NW,SE,SW = range(8)

# Pawn push and capture tables
W_PUSH1 = tuple((sq + 8) if sq < 56 else -1 for sq in range(64))
B_PUSH1 = tuple((sq - 8) if sq >= 8 else -1 for sq in range(64))

W_PUSH2 = tuple((sq + 16) if 8 <= sq < 16 else -1 for sq in range(64))    # rank 2
B_PUSH2 = tuple((sq - 16) if 48 <= sq < 56 else -1 for sq in range(64))   # rank 7

W_CAPS = tuple(
    tuple(dst for dst in (
        (sq + 7 if (sq & 7) != 0 else -1),
        (sq + 9 if (sq & 7) != 7 else -1),
    ) if 0 <= dst < 64)
    for sq in range(64)
)

B_CAPS = tuple(
    tuple(dst for dst in (
        (sq - 9 if (sq & 7) != 0 else -1),
        (sq - 7 if (sq & 7) != 7 else -1),
    ) if 0 <= dst < 64)
    for sq in range(64)
)

########################################
### End of precomputed attack tables ###
########################################

# material count indices: wp,wn,wb,wr,wq,bp,bn,bb,br,bq
MAT_INDEX = [-1] * (_MAX_PC + 1)
MAT_INDEX[p.WHITE_PAWN]   = 0
MAT_INDEX[p.WHITE_KNIGHT] = 1
MAT_INDEX[p.WHITE_BISHOP] = 2
MAT_INDEX[p.WHITE_ROOK]   = 3
MAT_INDEX[p.WHITE_QUEEN]  = 4
MAT_INDEX[p.BLACK_PAWN]   = 5
MAT_INDEX[p.BLACK_KNIGHT] = 6
MAT_INDEX[p.BLACK_BISHOP] = 7
MAT_INDEX[p.BLACK_ROOK]   = 8
MAT_INDEX[p.BLACK_QUEEN]  = 9
# kings and NONE remain -1


@dataclass(slots=True)
class Undo:
    move: Move
    moved_piece: int
    captured_piece: int
    captured_square: int              # usually dst; for en passant it's behind dst
    prev_castling_rights: int
    prev_en_passant_target: int | None
    prev_halfmove_clock: int
    prev_is_white_to_move: bool
    prev_white_king_sq: int
    prev_black_king_sq: int
    prev_zkey:int
    rook_src: int = -1                # for castling
    rook_dst: int = -1
    
@dataclass(frozen=True, slots=True)
class BoardState:
    board: bytes
    is_white_to_move: bool
    castling_rights: int
    en_passant_target: int | None
    halfmove_clock: int
    fullmove_number: int
    white_king_sq: int
    black_king_sq: int
    zkey: int
    rep_counts: tuple[tuple[int, int], ...]
    rep_stack: tuple[int, ...]

# For Zobrist hashing
_ZRAND = random.Random(0xC0FFEE1234)  # deterministic

def _rand64() -> int:
    return _ZRAND.getrandbits(64)

# Zobrist hashing
_Z_PIECES = (
    p.WHITE_PAWN, p.WHITE_KNIGHT, p.WHITE_BISHOP, p.WHITE_ROOK, p.WHITE_QUEEN, p.WHITE_KING,
    p.BLACK_PAWN, p.BLACK_KNIGHT, p.BLACK_BISHOP, p.BLACK_ROOK, p.BLACK_QUEEN, p.BLACK_KING,
)
_ZPI      = {pc: i for i, pc in enumerate(_Z_PIECES)}
_Z_PIECE  = [[_rand64() for _ in range(64)] for _ in range(len(_Z_PIECES))]
_Z_CASTLE = [_rand64() for _ in range(16)]  # index by castling_rights bitmask (0..15)
_Z_EPFILE = [_rand64() for _ in range(8)]
_Z_SIDE   = _rand64()


def _file(sq: int) -> int: return sq & 7
def _rank(sq: int) -> int: return sq >> 3
def _idx(f: int, r: int) -> int: return f + (r << 3)

class Board:

    NO_PIECE: int    = 0
    ENEMY_PIECE: int = -1
    OWN_PIECE: int   = 1
    
    WHITE_CASTLE_KINGSIDE: int  = 1<<0 
    WHITE_CASTLE_QUEENSIDE: int = 1<<1 
    BLACK_CASTLE_KINGSIDE: int  = 1<<2 
    BLACK_CASTLE_QUEENSIDE: int = 1<<3
    
    
    def __init__(self, ranks=8, files=8):
        if ranks != 8 or files != 8:
            raise ValueError("This engine currently supports only 8x8 chess.")
        
        self._ranks:     int = ranks
        self._files:     int = files
        self._grid_size: int = ranks * files
        self._white_king_sq: int = -1
        self._black_king_sq: int = -1


        self._board: bytearray = bytearray(self._grid_size)        
        self._castling_rights: int = ( self.WHITE_CASTLE_KINGSIDE 
                                     | self .WHITE_CASTLE_QUEENSIDE
                                     | self.BLACK_CASTLE_KINGSIDE 
                                     | self.BLACK_CASTLE_QUEENSIDE)

        self._en_passant_target: int | None = None
        
        self._is_white_to_move: bool = True
        self._halfmove_clock: int = 0 # for fifty-move rule
        

        self._zkey: int = 0
        
        # material count       
        self._mat = [0,0,0,0,0, 0,0,0,0,0]  # wp..bq

        # repetition tracking
        self._rep_counts: dict[int, int] = {}
        self._rep_stack: list[int] = []  # keys after each ply (or include initial)

        self.init_board()
        
    def init_board(self, fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") -> None:
        self.set_fen(fen)
        self._zkey = self._recompute_zobrist()
        self._rep_counts = {self._zkey: 1}
        self._rep_stack = [self._zkey]
        
    def make_move(self, move: Move) -> bool:
        src = move.src_square
        if not (0 <= src < self._grid_size):
            return False

        # IMPORTANT: require exact move match (flags included)
        if move not in self.get_legal_moves(src):
            return False

        self._do_move(move)   # use do/undo implementation
        return True
        
    def _clear_castling_rights(self, mask: int) -> None:
        self._castling_rights &= ~mask

    def _has_castling_rights(self, mask: int) -> bool:
        return (self._castling_rights & mask) != 0
    
    # def idx_to_f_r(self, idx: int) -> tuple[int, int]:
    #     r: int = idx // self._files
    #     f: int = idx % self._files
    #     return f, r

    # def get_idx(self, f: int, r: int) -> int:
    #     return f + (r * self._files)

    def _gen_pawn_moves(self, src: int) -> list[Move]:
        board = self._board
        piece = board[src]
        white = IS_WHITE[piece]

        moves: list[Move] = []
        append = moves.append
        ep = self._en_passant_target

        if white:
            one = W_PUSH1[src]
            if one != -1 and board[one] == p.NONE:
                if one >= 56:
                    for to in (PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN):
                        append(Move.promotion_to(src, one, to, 0))
                else:
                    append(Move.normal(src, one, 0))
                    two = W_PUSH2[src]
                    if two != -1 and board[two] == p.NONE:
                        append(Move.double_pawn(src, two))

            for dst in W_CAPS[src]:
                tp = board[dst]
                if tp != p.NONE and not IS_WHITE[tp]:
                    if dst >= 56:
                        for to in (PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN):
                            append(Move.promotion_to(src, dst, to, tp))
                    else:
                        append(Move.normal(src, dst, tp))
                elif ep is not None and dst == ep:
                    # IMPORTANT: keep captured_piece set so exact-match move validation works
                    append(Move(src, dst, F_EN_PASSANT | F_CAPTURE, PROMO_NONE, p.BLACK_PAWN))
        else:
            one = B_PUSH1[src]
            if one != -1 and board[one] == p.NONE:
                if one < 8:
                    for to in (PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN):
                        append(Move.promotion_to(src, one, to, 0))
                else:
                    append(Move.normal(src, one, 0))
                    two = B_PUSH2[src]
                    if two != -1 and board[two] == p.NONE:
                        append(Move.double_pawn(src, two))

            for dst in B_CAPS[src]:
                tp = board[dst]
                if tp != p.NONE and IS_WHITE[tp]:
                    if dst < 8:
                        for to in (PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN):
                            append(Move.promotion_to(src, dst, to, tp))
                    else:
                        append(Move.normal(src, dst, tp))
                elif ep is not None and dst == ep:
                    append(Move(src, dst, F_EN_PASSANT | F_CAPTURE, PROMO_NONE, p.WHITE_PAWN))

        return moves

    def _gen_king_moves(self, src: int) -> list[Move]:
        board = self._board
        isw = IS_WHITE
        white = isw[board[src]]

        moves: list[Move] = []
        append = moves.append

        # normal king steps
        for dst in KING_MOVES[src]:
            tp = board[dst]
            if tp == p.NONE:
                append(Move.normal(src, dst, 0))
            elif isw[tp] != white:
                append(Move.normal(src, dst, tp))

        # castling (obstructions only)
        # e1=4, e8=60
        start = 4 if white else 60
        if src != start:
            return moves

        rook_piece = p.WHITE_ROOK if white else p.BLACK_ROOK

        if white:
            # white queenside: rook a1=0, empty squares b1=1 c1=2 d1=3
            if self._has_castling_rights(self.WHITE_CASTLE_QUEENSIDE):
                if board[0] == rook_piece and board[1] == p.NONE and board[2] == p.NONE and board[3] == p.NONE:
                    append(Move.castle(src, 2))  # c1

            # white kingside: rook h1=7, empty squares f1=5 g1=6
            if self._has_castling_rights(self.WHITE_CASTLE_KINGSIDE):
                if board[7] == rook_piece and board[5] == p.NONE and board[6] == p.NONE:
                    append(Move.castle(src, 6))  # g1
        else:
            # black queenside: rook a8=56, empty squares b8=57 c8=58 d8=59
            if self._has_castling_rights(self.BLACK_CASTLE_QUEENSIDE):
                if board[56] == rook_piece and board[57] == p.NONE and board[58] == p.NONE and board[59] == p.NONE:
                    append(Move.castle(src, 58))  # c8

            # black kingside: rook h8=63, empty squares f8=61 g8=62
            if self._has_castling_rights(self.BLACK_CASTLE_KINGSIDE):
                if board[63] == rook_piece and board[61] == p.NONE and board[62] == p.NONE:
                    append(Move.castle(src, 62))  # g8

        return moves

    def _is_enemy(self, piece: int, white_to_move: bool) -> bool:
        if piece == p.NONE:
            return False
        return IS_WHITE[piece] != white_to_move

    def _gen_knight_moves(self, src: int) -> list[Move]:
        board = self._board
        piece = board[src]
        white = IS_WHITE[piece]
        moves: list[Move] = []
        for dst in KNIGHT_MOVES[src]:
            tp = board[dst]
            if tp == p.NONE:
                moves.append(Move.normal(src, dst, 0))
            elif IS_WHITE[tp] != white:
                moves.append(Move.normal(src, dst, tp))
        return moves

    def _gen_bishop_moves(self, src: int) -> list[Move]:
        board = self._board
        isw = IS_WHITE
        white = isw[board[src]]

        moves: list[Move] = []
        append = moves.append

        for dir_i in (NE, NW, SE, SW):
            for dst in RAYS[src][dir_i]:
                tp = board[dst]
                if tp == p.NONE:
                    append(Move.normal(src, dst, 0))
                    continue
                if isw[tp] != white:
                    append(Move.normal(src, dst, tp))
                break

        return moves

    def _gen_rook_moves(self, src: int) -> list[Move]:
        board = self._board
        white = IS_WHITE[board[src]]
        moves: list[Move] = []
        for dir_i in (E, W, N, S):
            for dst in RAYS[src][dir_i]:
                tp = board[dst]
                if tp == p.NONE:
                    moves.append(Move.normal(src, dst, 0))
                    continue
                if IS_WHITE[tp] != white:
                    moves.append(Move.normal(src, dst, tp))
                break
        return moves

    def _gen_queen_moves(self, src: int) -> list[Move]:
        board = self._board
        isw = IS_WHITE
        white = isw[board[src]]

        moves: list[Move] = []
        append = moves.append

        # all 8 directions
        for dir_i in range(8):
            for dst in RAYS[src][dir_i]:
                tp = board[dst]
                if tp == p.NONE:
                    append(Move.normal(src, dst, 0))
                    continue
                if isw[tp] != white:
                    append(Move.normal(src, dst, tp))
                break

        return moves
    
    def get_pseudolegal_moves(self, src: int) -> list[Move]:
        piece: int = self._board[src]
        
        if piece == p.NONE or self._is_enemy(piece, self._is_white_to_move):
            return []
        
        match PTYPE[piece]:
            case p.PAWN:
                return self._gen_pawn_moves(src)
            
            case p.KNIGHT:
                return self._gen_knight_moves(src)
            
            case p.BISHOP:
                return self._gen_bishop_moves(src)
            
            case p.ROOK:
                return self._gen_rook_moves(src)
            
            case p.QUEEN:
                return self._gen_queen_moves(src)
            
            case p.KING:
                return self._gen_king_moves(src)
            
        return []
    
    def get_legal_moves(self, src: int) -> list[Move]:
        pseudo = self.get_pseudolegal_moves(src)
        if not pseudo:
            return []

        side = self._is_white_to_move  # mover color BEFORE move
        king_sq0 = self._white_king_sq if side else self._black_king_sq
        if king_sq0 == -1:
            return []

        legal: list[Move] = []

        for m in pseudo:
            # Castling extra restriction: not out of / through / into check
            if m.flags & F_CASTLE:
                if self.is_square_attacked(king_sq0, by_white=not side):
                    continue
                # f_src, r_src = _file(m.src_square), _rank(m.src_square)
                f_src: int = m.src_square & 7
                r_src: int = m.src_square >> 3
                # f_dst, _     = _file(m.dst_square), _rank(m.dst_square)
                f_dst: int = m.dst_square & 7
                if f_dst > f_src:  # kingside
                    # through = [_idx(f_src + 1, r_src), _idx(f_src + 2, r_src)]
                    through = [f_src + 1 + (r_src << 3), f_src + 2 + (r_src << 3)]
                else:              # queenside
                    # through = [_idx(f_src - 1, r_src), _idx(f_src - 2, r_src)]
                    through = [f_src - 1 + (r_src << 3), f_src - 2 + (r_src << 3)]
                if any(self.is_square_attacked(sq, by_white=not side) for sq in through):
                    continue

            undo = self._do_move(m)
            king_sq = self._white_king_sq if side else self._black_king_sq  # mover’s king square after move
            illegal = (king_sq == -1) or self.is_square_attacked(king_sq, by_white=not side)
            self._undo_move(undo)

            if not illegal:
                legal.append(m)

        return legal
    
    def get_board(self) -> Sequence[int]:
        return self._board

    def get_is_white_to_move(self) -> bool:
        return self._is_white_to_move
    
    def is_square_attacked(self, square: int, by_white: bool) -> bool:
        board = self._board
        isw = IS_WHITE
        ptype = PTYPE

        # pawn
        if by_white:
            wp = p.WHITE_PAWN
            for s in PAWN_ATTACKERS_WHITE[square]:
                if board[s] == wp:
                    return True
        else:
            bp = p.BLACK_PAWN
            for s in PAWN_ATTACKERS_BLACK[square]:
                if board[s] == bp:
                    return True

        # knight
        kn = p.WHITE_KNIGHT if by_white else p.BLACK_KNIGHT
        for s in KNIGHT_MOVES[square]:
            if board[s] == kn:
                return True

        # king
        kk = p.WHITE_KING if by_white else p.BLACK_KING
        for s in KING_MOVES[square]:
            if board[s] == kk:
                return True

        # sliders
        # diagonals: bishop/queen
        for dir_i in (NE, NW, SE, SW):
            for s in RAYS[square][dir_i]:
                pc = board[s]
                if pc == p.NONE:
                    continue
                if isw[pc] == by_white:
                    t = ptype[pc]
                    if t == p.BISHOP or t == p.QUEEN:
                        return True
                break

        # orthogonals: rook/queen
        for dir_i in (E, W, N, S):
            for s in RAYS[square][dir_i]:
                pc = board[s]
                if pc == p.NONE:
                    continue
                if isw[pc] == by_white:
                    t = ptype[pc]
                    if t == p.ROOK or t == p.QUEEN:
                        return True
                break

        return False
    
    def _find_king(self, white: bool) -> int:
        return self._white_king_sq if white else self._black_king_sq


    def _do_move(self, move: Move) -> Undo:
        src, dst = move.src_square, move.dst_square
        moved_piece = int(self._board[src])
        captured_piece = int(self._board[dst])
        captured_square = dst

        undo = Undo(
            move=move,
            moved_piece=moved_piece,
            captured_piece=captured_piece,
            captured_square=captured_square,
            prev_castling_rights=self._castling_rights,
            prev_en_passant_target=self._en_passant_target,
            prev_halfmove_clock=self._halfmove_clock,
            prev_is_white_to_move=self._is_white_to_move,
            prev_white_king_sq=self._white_king_sq,
            prev_black_king_sq=self._black_king_sq,
            prev_zkey=self._zkey,
        )

        # --- Zobrist: remove old EP / old castling / old side ---
        old_epf = self._ep_file_for_hash()
        if old_epf is not None:
            self._zkey ^= _Z_EPFILE[old_epf]

        self._zkey ^= _Z_CASTLE[self._castling_rights & 0xF]

        # side to move toggles every ply
        self._zkey ^= _Z_SIDE

        # reset en passant by default
        self._en_passant_target = None

        # halfmove clock (for 50-move draw)
        # reset on pawn move or any capture
        if PTYPE[moved_piece] == p.PAWN or move.flags & F_CAPTURE:
            self._halfmove_clock = 0
        else:
            self._halfmove_clock += 1
 
        # --- Zobrist: remove moved piece from src ---           
        self._z_xor_piece(moved_piece, src)

        # --- special moves ---
        if move.flags & F_EN_PASSANT:
            # capture pawn behind destination
            if IS_WHITE[moved_piece]:
                cap_sq = dst - self._files
            else:
                cap_sq = dst + self._files
            cap_piece = int(self._board[cap_sq])

            if cap_piece == p.NONE:
                raise AssertionError(
                    f"Invalid en-passant move: no pawn on capture square "
                    f"move={move}, cap_sq={cap_sq}, fen={self.to_fen()}"
                )
            
            self._mat_dec(cap_piece)
            self._z_xor_piece(cap_piece, cap_sq)
            undo.captured_square = cap_sq
            undo.captured_piece = cap_piece
            self._board[cap_sq] = p.NONE
        else:
            if captured_piece != p.NONE:
                self._mat_dec(captured_piece)
                self._z_xor_piece(captured_piece, dst) 
            
        if move.flags & F_CASTLE:
            # f_src, r_src = _file(src), _rank(src)
            f_src: int = src & 7
            r_src: int = src >> 3
            # f_dst        = _file(dst)
            f_dst: int = dst & 7
            # kingside: dst file is 6; queenside: dst file is 2
            if f_dst > f_src:
                # rook_src = _idx(self._files - 1, r_src)
                rook_src = (self._files - 1) + (r_src << 3)
                # rook_dst = _idx(self._files - 3, r_src)
                rook_dst = (self._files - 3) + (r_src << 3)
            else:
                # rook_src = _idx(0, r_src)
                rook_src = 0 + (r_src << 3)
                # rook_dst = _idx(3, r_src)
                rook_dst = 3 + (r_src << 3)
            undo.rook_src, undo.rook_dst = rook_src, rook_dst
            self._board[rook_dst] = self._board[rook_src]
            self._board[rook_src] = p.NONE
            rook_piece = p.WHITE_ROOK if IS_WHITE[moved_piece] else p.BLACK_ROOK
            self._z_xor_piece(rook_piece, rook_src)
            self._z_xor_piece(rook_piece, rook_dst)

        # update castling rights due to king/rook moves or rook capture
        t = PTYPE[moved_piece]
        if t == p.KING:
            if IS_WHITE[moved_piece]:
                self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE | self.WHITE_CASTLE_QUEENSIDE)
            else:
                self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE | self.BLACK_CASTLE_QUEENSIDE)
        elif t == p.ROOK:
            # f_src, r_src = _file(src), _rank(src)
            f_src: int = src & 7
            r_src: int = src >> 3
            if IS_WHITE[moved_piece]:
                if f_src == 0 and r_src == 0: self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                if f_src == 7 and r_src == 0: self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)
            else:
                if f_src == 0 and r_src == 7: self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                if f_src == 7 and r_src == 7: self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)

        if PTYPE[undo.captured_piece] == p.ROOK:
            # f_dst, r_dst = _file(dst), _rank(dst)
            f_dst: int = dst & 7
            r_dst: int = dst >> 3
            if IS_WHITE[undo.captured_piece]:
                if f_dst == 0 and r_dst == 0: self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                if f_dst == 7 and r_dst == 0: self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)
            else:
                if f_dst == 0 and r_dst == 7: self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                if f_dst == 7 and r_dst == 7: self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)

        # double pawn sets en passant target
        if move.flags & F_DOUBLE_PAWN:
            # f_src, r_src = _file(src), _rank(src)
            f_src: int = src & 7
            r_src: int = src >> 3
            # _, r_dst = _file(dst), _rank(dst)
            r_dst: int = dst >> 3
            # self._en_passant_target = _idx(f_src, (r_src + r_dst) // 2)
            self._en_passant_target = f_src + (((r_src + r_dst) // 2) << 3)

        # apply main piece move
        self._board[dst] = self._board[src]
        self._board[src] = p.NONE
 
        # update king square       
        if PTYPE[moved_piece] == p.KING:
            if IS_WHITE[moved_piece]:
                self._white_king_sq = dst
            else:
                self._black_king_sq = dst

        # promotion replaces piece on dst
        if move.flags & F_PROMOTION:
            self._mat_dec(moved_piece)
            color = p.piece_color(moved_piece)
            promo_t = {
                PROMO_QUEEN:  p.QUEEN,
                PROMO_ROOK:   p.ROOK,
                PROMO_BISHOP: p.BISHOP,
                PROMO_KNIGHT: p.KNIGHT,
            }.get(move.promotion)
            if promo_t is None:
                raise AssertionError("Promotion flag set but invalid promotion piece")
            promo_piece = p.make_piece(promo_t, color)
            self._mat_inc(promo_piece)
            self._board[dst] = promo_piece
            self._z_xor_piece(promo_piece, dst)
        else:
            self._z_xor_piece(moved_piece, dst)

        # flip side FIRST so EP hashing uses the correct side-to-move
        self._is_white_to_move = not self._is_white_to_move

        # --- Zobrist: add new castling / new EP (must use new side) ---
        self._zkey ^= _Z_CASTLE[self._castling_rights & 0xF]
        new_epf = self._ep_file_for_hash()  # now uses the flipped side
        if new_epf is not None:
            self._zkey ^= _Z_EPFILE[new_epf]

        # record position repetition (now the key matches the real position)
        self._rep_stack.append(self._zkey)
        self._rep_counts[self._zkey] = self._rep_counts.get(self._zkey, 0) + 1

        if DEBUG_MAT:
            self._debug_assert_mat_ok()

        return undo


    def _undo_move(self, undo: Undo) -> None:
        move = undo.move
        src, dst = move.src_square, move.dst_square

        # repetition bookkeeping: we are leaving current position
        cur = self._rep_stack.pop()
        cnt = self._rep_counts[cur] - 1
        if cnt == 0:
            del self._rep_counts[cur]
        else:
            self._rep_counts[cur] = cnt

        # restore turn first (so helpers that depend on side can be sane)
        self._is_white_to_move = undo.prev_is_white_to_move
        
        # reverse promotion material change
        if move.flags & F_PROMOTION:
            color = p.piece_color(undo.moved_piece)  # pawn color
            promo_t = {
                PROMO_QUEEN:  p.QUEEN,
                PROMO_ROOK:   p.ROOK,
                PROMO_BISHOP: p.BISHOP,
                PROMO_KNIGHT: p.KNIGHT,
            }[move.promotion]
            promo_piece = p.make_piece(promo_t, color)
            self._mat_dec(promo_piece)       # remove promoted piece
            self._mat_inc(undo.moved_piece)  # add pawn back

        # restore captured piece material (works for normal + en passant)
        if undo.captured_piece != p.NONE:
            self._mat_inc(undo.captured_piece)

        # restore clocks/state
        self._castling_rights = undo.prev_castling_rights
        self._en_passant_target = undo.prev_en_passant_target
        self._halfmove_clock = undo.prev_halfmove_clock

        # undo promotion: put original pawn back on src
        # (we stored moved_piece which is the original piece before promotion)
        self._board[src] = undo.moved_piece

        # restore captured piece
        self._board[undo.captured_square] = undo.captured_piece

        # clear dst unless captured_square==dst (normal capture)
        if undo.captured_square != dst:
            self._board[dst] = p.NONE

        # undo castling rook move
        if move.flags & F_CASTLE:
            self._board[undo.rook_src] = self._board[undo.rook_dst]
            self._board[undo.rook_dst] = p.NONE
            
        # restore king square
        self._white_king_sq = undo.prev_white_king_sq
        self._black_king_sq = undo.prev_black_king_sq
        
        # recompute Zobrist key
        self._zkey = undo.prev_zkey

        if DEBUG_MAT:
            self._debug_assert_mat_ok()

    def get_all_legal_moves(self) -> list[Move]:
        board = self._board
        get_legal_moves = self.get_legal_moves
        grid_size = self._grid_size
        
        
        legal: list[Move] = []
        side = self._is_white_to_move
        for src in range(grid_size):
            piece = int(board[src])
            if piece == p.NONE:
                continue
            if IS_WHITE[piece] != side:
                continue
            legal.extend(get_legal_moves(src))
        return legal

    def in_check(self, white: bool) -> bool:
        k = self._white_king_sq if white else self._black_king_sq
        return k != -1 and self.is_square_attacked(k, by_white=not white)

    def game_end_state(self) -> tuple[bool, str]:
        """
        (done, reason) where reason in {
        'checkmate','stalemate','threefold repetition','fifty-move rule',
        'insufficient material','playing'
        }.
        """
        moves = self.get_all_legal_moves()

        # No legal moves -> mate or stalemate (these must override everything)
        if not moves:
            side = self._is_white_to_move
            if self.in_check(side):
                return True, "checkmate"
            return True, "stalemate"

        # Draws that can happen while moves still exist
        if self.is_insufficient_material():
            return True, "insufficient material"

        if self._halfmove_clock >= 100:
            return True, "fifty-move rule"

        if self.is_threefold_repetition():
            return True, "threefold repetition"

        return False, "playing"

    _FEN_TO_PIECE = {
        "P": p.WHITE_PAWN,   "N": p.WHITE_KNIGHT, "B": p.WHITE_BISHOP,
        "R": p.WHITE_ROOK,   "Q": p.WHITE_QUEEN,  "K": p.WHITE_KING,
        "p": p.BLACK_PAWN,   "n": p.BLACK_KNIGHT, "b": p.BLACK_BISHOP,
        "r": p.BLACK_ROOK,   "q": p.BLACK_QUEEN,  "k": p.BLACK_KING,
    }

    _PIECE_TO_FEN = {v: k for k, v in _FEN_TO_PIECE.items()}

    def algebraic_to_idx(self, sq: str) -> int:
        # "a1" -> 0, "h8" -> 63
        f = ord(sq[0]) - ord("a")
        r = int(sq[1]) - 1
        if not (0 <= f < 8 and 0 <= r < 8):
            raise ValueError(f"Bad square: {sq}")
        # return _idx(f, r)
        return f + (r << 3)

    def idx_to_algebraic(self, idx: int) -> str:
        # f, r = _file(idx), _rank(idx)
        f = idx & 7
        r = idx >> 3
        return f"{chr(ord('a') + f)}{r + 1}"

    def set_fen(self, fen: str) -> None:
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Bad FEN: {fen}")

        placement, active, castling, ep = parts[:4]
        halfmove = int(parts[4]) if len(parts) >= 5 else 0
        fullmove = int(parts[5]) if len(parts) >= 6 else 1  # optional (but nice to keep)
        
        self._mat = [0,0,0,0,0, 0,0,0,0,0]  # reset material count

        self._reset_board()
        self._white_king_sq = -1
        self._black_king_sq = -1

        rows = placement.split("/")
        if len(rows) != 8:
            raise ValueError(f"Bad FEN placement rows: {placement}")

        for fen_r, row in enumerate(rows):          # fen_r=0 is rank 8
            r = 7 - fen_r                           # engine rank 7 is rank 8
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                    continue
                if ch not in self._FEN_TO_PIECE:
                    raise ValueError(f"Bad FEN char: {ch}")
                if f >= 8:
                    raise ValueError("Bad FEN row overflow")
                # idx = _idx(f, r)
                idx = f + (r << 3)
                piece = self._FEN_TO_PIECE[ch]
                self._board[idx] = piece
                self._mat_inc(piece)
                if piece == p.WHITE_KING:
                    self._white_king_sq = idx
                elif piece == p.BLACK_KING:
                    self._black_king_sq = idx
                f += 1
            if f != 8:
                raise ValueError(f"Bad FEN row width: {row}")

        self._is_white_to_move = (active == "w")

        rights = 0
        if castling != "-":
            if "K" in castling: rights |= self.WHITE_CASTLE_KINGSIDE
            if "Q" in castling: rights |= self.WHITE_CASTLE_QUEENSIDE
            if "k" in castling: rights |= self.BLACK_CASTLE_KINGSIDE
            if "q" in castling: rights |= self.BLACK_CASTLE_QUEENSIDE
        self._castling_rights = rights

        # Per FEN, ep can be set even if no capture is possible. :contentReference[oaicite:1]{index=1}
        self._en_passant_target = None if ep == "-" else self.algebraic_to_idx(ep)

        self._halfmove_clock = halfmove
        self._fullmove_number = fullmove  # add this attribute in __init__ if you want

        if self._white_king_sq == -1 or self._black_king_sq == -1:
            raise ValueError("FEN must contain both kings")

    def to_fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            empties = 0
            out = []
            for f in range(8):
                # idx = _idx(f, r)
                idx = f + (r << 3)
                piece = int(self._board[idx])
                if piece == p.NONE:
                    empties += 1
                else:
                    if empties:
                        out.append(str(empties))
                        empties = 0
                    out.append(self._PIECE_TO_FEN.get(piece, "?"))
            if empties:
                out.append(str(empties))
            rows.append("".join(out))

        active = "w" if self._is_white_to_move else "b"

        c = []
        if self._castling_rights & self.WHITE_CASTLE_KINGSIDE:  c.append("K")
        if self._castling_rights & self.WHITE_CASTLE_QUEENSIDE: c.append("Q")
        if self._castling_rights & self.BLACK_CASTLE_KINGSIDE:  c.append("k")
        if self._castling_rights & self.BLACK_CASTLE_QUEENSIDE: c.append("q")
        castling = "".join(c) if c else "-"

        ep = "-" if self._en_passant_target is None else self.idx_to_algebraic(self._en_passant_target)

        half = str(self._halfmove_clock)
        full = str(getattr(self, "_fullmove_number", 1))
        return f"{'/'.join(rows)} {active} {castling} {ep} {half} {full}"

    def _reset_board(self) -> None:
        self._board[:] = b"\x00" * self._grid_size
 

    def _ep_file_for_hash(self) -> int | None:
        """
        Return ep file (0..7) *only if* an en-passant capture is legal in principle,
        else None. This matches FIDE repetition rules and common engine practice.
        """
        ep = self._en_passant_target
        if ep is None:
            return None

        # f_ep, r_ep = _file(ep), _rank(ep)
        f_ep: int = ep & 7
        r_ep: int = ep >> 3

        # Side to move must have a pawn that can capture onto ep square.
        if self._is_white_to_move:
            # white captures upward (from rank-1 to rank)
            r_from = r_ep - 1
            if r_from < 0:
                return None
            for df in (-1, 1):
                f_from = f_ep + df
                if 0 <= f_from < 8:
                    # src = _idx(f_from, r_from)
                    src = f_from + (r_from << 3)
                    if self._board[src] == p.WHITE_PAWN:
                        return f_ep
        else:
            # black captures downward (from rank+1 to rank)
            r_from = r_ep + 1
            if r_from > 7:
                return None
            for df in (-1, 1):
                f_from = f_ep + df
                if 0 <= f_from < 8:
                    # src = _idx(f_from, r_from)
                    src = f_from + (r_from << 3)
                    if self._board[src] == p.BLACK_PAWN:
                        return f_ep

        return None

    def _recompute_zobrist(self) -> int:
        z = 0
        board = self._board

        for sq in range(self._grid_size):
            pc = int(board[sq])
            if pc != p.NONE:
                z ^= _Z_PIECE[_ZPI[pc]][sq]

        z ^= _Z_CASTLE[self._castling_rights & 0xF]

        epf = self._ep_file_for_hash()
        if epf is not None:
            z ^= _Z_EPFILE[epf]
        if self._is_white_to_move:
            z ^= _Z_SIDE  # convention: side key means "white to move"
        return z

    def _z_xor_piece(self, piece: int, sq: int) -> None:
        self._zkey ^= _Z_PIECE[_ZPI[piece]][sq]

    def is_threefold_repetition(self) -> bool:
        return self._rep_counts.get(self._zkey, 0) >= 3

    def is_insufficient_material(self) -> bool:
        """
        Returns True if neither side can possibly checkmate with the material on the board.
        Common minimal rules:
        - K vs K
        - K+N vs K
        - K+B vs K
        - K+N vs K+N
        - K+NN vs K (and symmetric)
        - K+B vs K+B with bishops on the same color complex
        - Any position with pawns/rooks/queens is NOT insufficient
        """
        knights = 0
        bishops = 0
        bishop_colors: list[int] = []

        for sq in range(self._grid_size):
            pc = int(self._board[sq])
            if pc == p.NONE:
                continue

            t = PTYPE[pc]
            if t == p.KING:
                continue

            if t in (p.PAWN, p.ROOK, p.QUEEN):
                return False

            if t == p.KNIGHT:
                knights += 1
            elif t == p.BISHOP:
                bishops += 1
                # f, r = _file(sq), _rank(sq)
                f: int = sq & 7
                r: int = sq >> 3
                bishop_colors.append((f + r) & 1)
            else:
                # should not happen in normal chess, but be conservative
                return False

        minors = knights + bishops

        # Bare kings
        if minors == 0:
            return True

        # Single minor piece total: K+N vs K or K+B vs K
        if minors == 1:
            return True

        # Knights only: up to 2 total knights cannot mate
        if bishops == 0 and knights <= 2:
            return True

        # Bishops only: if all bishops are on the same color complex (common cases: K+B vs K+B same color)
        if knights == 0 and bishops >= 1:
            if all(c == bishop_colors[0] for c in bishop_colors):
                return True

        return False

    def get_state(self) -> "BoardState":
        return BoardState(
            board=bytes(self._board),
            is_white_to_move=self._is_white_to_move,
            castling_rights=self._castling_rights,
            en_passant_target=self._en_passant_target,
            halfmove_clock=self._halfmove_clock,
            fullmove_number=getattr(self, "_fullmove_number", 1),
            white_king_sq=self._white_king_sq,
            black_king_sq=self._black_king_sq,
            zkey=self._zkey,
            rep_counts=tuple(self._rep_counts.items()),
            rep_stack=tuple(self._rep_stack),
        )

    def set_state(self, s: "BoardState") -> None:
        self._board[:] = s.board
        self._is_white_to_move = s.is_white_to_move
        self._castling_rights = s.castling_rights
        self._en_passant_target = s.en_passant_target
        self._halfmove_clock = s.halfmove_clock
        self._fullmove_number = s.fullmove_number
        self._white_king_sq = s.white_king_sq
        self._black_king_sq = s.black_king_sq
        self._zkey = s.zkey

        self._recompute_mat()

        # Make do/undo safe after restoring
        self._rep_counts = dict(s.rep_counts)
        self._rep_stack = list(s.rep_stack)
        
    def _mat_inc(self, pc: int) -> None:
        i = MAT_INDEX[pc]
        if i != -1:
            self._mat[i] += 1

    def _mat_dec(self, pc: int) -> None:
        i = MAT_INDEX[pc]
        if i != -1:
            self._mat[i] -= 1

    def _recompute_mat(self) -> None:
        m = [0,0,0,0,0, 0,0,0,0,0]
        for pc in self._board:
            i = MAT_INDEX[int(pc)]
            if i != -1:
                m[i] += 1
        self._mat = m
        
    def _debug_assert_mat_ok(self) -> None:
        m = [0,0,0,0,0, 0,0,0,0,0]
        for pc in self._board:
            i = MAT_INDEX[int(pc)]
            if i != -1:
                m[i] += 1
        assert m == self._mat, (self._mat, m)

    def current_repetition_count(self) -> int:
        """
        Return how many times the current position key has occurred in the current game history.

        This uses the engine's internal repetition key. It should include the same
        information that your threefold-repetition logic already uses.
        """
        return int(self._rep_counts.get(self._zkey, 0))


    def repetition_count_after(self, move: Move) -> int:
        """
        Return the repetition count that would hold after making `move`,
        without permanently modifying the board.
        """
        undo = self._do_move(move)
        try:
            return self.current_repetition_count()
        finally:
            self._undo_move(undo)


    def is_repetition_risk_after(self, move: Move, *, threshold: int = 2) -> bool:
        """
        True if making this move reaches a position that has occurred at least
        `threshold` times.

        threshold=2 means "this is already a repeated position / approaching danger".
        threshold=3 means "this reaches threefold repetition" under your engine logic.
        """
        return self.repetition_count_after(move) >= threshold


    def is_threefold_repetition_after(self, move: Move) -> bool:
        """
        True if making this move would reach a position with repetition count >= 3.
        """
        return self.repetition_count_after(move) >= 3

    @contextmanager
    def temporary_move(self, move: Move) -> Iterator[None]:
        """
        Temporarily make a move and automatically undo it.

        Usage:
            with board.temporary_move(move):
                ...
        """
        undo = self._do_move(move)
        try:
            yield
        finally:
            self._undo_move(undo)

    def get_all_legal_moves_for_side(self, white: bool) -> list[Move]:
        """
        Return all legal moves for the requested side without permanently changing turn.

        Important:
        En-passant is only legal for the actual side to move. If we are querying
        the non-side-to-move for static mobility/evaluation purposes, we must
        disable the EP target; otherwise bogus en-passant moves can be generated.
        """
        old_turn = self._is_white_to_move
        old_ep = self._en_passant_target
        old_zkey = self._zkey

        try:
            self._is_white_to_move = white

            if white != old_turn:
                self._en_passant_target = None

            # Keep Zobrist internally consistent while get_legal_moves()
            # temporarily makes/undoes moves.
            self._zkey = self._recompute_zobrist()

            return self.get_all_legal_moves()

        finally:
            self._is_white_to_move = old_turn
            self._en_passant_target = old_ep
            self._zkey = old_zkey


    def legal_mobility(self, white: bool) -> int:
        """
        Number of legal moves available to the requested side.
        """
        return len(self.get_all_legal_moves_for_side(white))

    def count_attackers(
        self,
        square: int,
        *,
        by_white: bool,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> int:
        """
        Count how many pieces of the given color attack `square`.

        This is a static pseudo-attack count, not a legal-move count.

        It includes:
          - pawn attacks, not pawn pushes
          - attacks/defenses of own occupied squares
          - pinned pieces as attackers

        Optional `piece_types` can restrict the count, e.g.
          {p.PAWN}
          {p.KNIGHT, p.BISHOP}
          {p.QUEEN}
        """
        if not (0 <= square < 64):
            raise ValueError(f"bad square index: {square}")

        board = self._board
        isw = IS_WHITE
        ptype = PTYPE

        count = 0

        def allowed(pc: int) -> bool:
            return piece_types is None or ptype[pc] in piece_types

        # Pawns.
        if by_white:
            wp = p.WHITE_PAWN
            for s in PAWN_ATTACKERS_WHITE[square]:
                if board[s] == wp and (piece_types is None or p.PAWN in piece_types):
                    count += 1
        else:
            bp = p.BLACK_PAWN
            for s in PAWN_ATTACKERS_BLACK[square]:
                if board[s] == bp and (piece_types is None or p.PAWN in piece_types):
                    count += 1

        # Knights.
        kn = p.WHITE_KNIGHT if by_white else p.BLACK_KNIGHT
        for s in KNIGHT_MOVES[square]:
            if board[s] == kn and (piece_types is None or p.KNIGHT in piece_types):
                count += 1

        # Kings.
        kk = p.WHITE_KING if by_white else p.BLACK_KING
        for s in KING_MOVES[square]:
            if board[s] == kk and (piece_types is None or p.KING in piece_types):
                count += 1

        # Diagonal sliders: bishop / queen.
        for dir_i in (NE, NW, SE, SW):
            for s in RAYS[square][dir_i]:
                pc = board[s]
                if pc == p.NONE:
                    continue

                if isw[pc] == by_white:
                    t = ptype[pc]
                    if (t == p.BISHOP or t == p.QUEEN) and allowed(pc):
                        count += 1

                break

        # Orthogonal sliders: rook / queen.
        for dir_i in (E, W, N, S):
            for s in RAYS[square][dir_i]:
                pc = board[s]
                if pc == p.NONE:
                    continue

                if isw[pc] == by_white:
                    t = ptype[pc]
                    if (t == p.ROOK or t == p.QUEEN) and allowed(pc):
                        count += 1

                break

        return count
    
    def attackers_to(
        self,
        square: int,
        *,
        by_white: bool,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> list[int]:
        """
        Return source squares of pieces of the given color attacking `square`.

        This is mainly useful for feature extraction/debugging.
        For fast all-board features, prefer attack_counts().
        """
        if not (0 <= square < 64):
            raise ValueError(f"bad square index: {square}")

        board = self._board
        isw = IS_WHITE
        ptype = PTYPE

        attackers: list[int] = []
        append = attackers.append

        def allowed(pc: int) -> bool:
            return piece_types is None or ptype[pc] in piece_types

        # Pawns.
        if by_white:
            wp = p.WHITE_PAWN
            for s in PAWN_ATTACKERS_WHITE[square]:
                if board[s] == wp and (piece_types is None or p.PAWN in piece_types):
                    append(s)
        else:
            bp = p.BLACK_PAWN
            for s in PAWN_ATTACKERS_BLACK[square]:
                if board[s] == bp and (piece_types is None or p.PAWN in piece_types):
                    append(s)

        # Knights.
        kn = p.WHITE_KNIGHT if by_white else p.BLACK_KNIGHT
        for s in KNIGHT_MOVES[square]:
            if board[s] == kn and (piece_types is None or p.KNIGHT in piece_types):
                append(s)

        # Kings.
        kk = p.WHITE_KING if by_white else p.BLACK_KING
        for s in KING_MOVES[square]:
            if board[s] == kk and (piece_types is None or p.KING in piece_types):
                append(s)

        # Diagonal sliders.
        for dir_i in (NE, NW, SE, SW):
            for s in RAYS[square][dir_i]:
                pc = board[s]
                if pc == p.NONE:
                    continue

                if isw[pc] == by_white:
                    t = ptype[pc]
                    if (t == p.BISHOP or t == p.QUEEN) and allowed(pc):
                        append(s)

                break

        # Orthogonal sliders.
        for dir_i in (E, W, N, S):
            for s in RAYS[square][dir_i]:
                pc = board[s]
                if pc == p.NONE:
                    continue

                if isw[pc] == by_white:
                    t = ptype[pc]
                    if (t == p.ROOK or t == p.QUEEN) and allowed(pc):
                        append(s)

                break

        return attackers
    
    def attack_counts(
        self,
        *,
        by_white: bool,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> tuple[int, ...]:
        """
        Return a 64-entry tuple where out[sq] is the number of pieces of
        `by_white` attacking sq.

        This is a static pseudo-attack map:
          - own occupied squares are counted as defended
          - slider attacks include the blocker square, then stop
          - pinned pieces still attack
        """
        board = self._board
        ptype = PTYPE

        counts = [0] * 64

        def type_allowed(t: int) -> bool:
            return piece_types is None or t in piece_types

        for src, pc_raw in enumerate(board):
            pc = int(pc_raw)

            if pc == p.NONE:
                continue

            if IS_WHITE[pc] != by_white:
                continue

            t = ptype[pc]

            if not type_allowed(t):
                continue

            if t == p.PAWN:
                targets = W_CAPS[src] if by_white else B_CAPS[src]
                for dst in targets:
                    counts[dst] += 1

            elif t == p.KNIGHT:
                for dst in KNIGHT_MOVES[src]:
                    counts[dst] += 1

            elif t == p.KING:
                for dst in KING_MOVES[src]:
                    counts[dst] += 1

            elif t == p.BISHOP:
                for dir_i in (NE, NW, SE, SW):
                    for dst in RAYS[src][dir_i]:
                        counts[dst] += 1
                        if board[dst] != p.NONE:
                            break

            elif t == p.ROOK:
                for dir_i in (E, W, N, S):
                    for dst in RAYS[src][dir_i]:
                        counts[dst] += 1
                        if board[dst] != p.NONE:
                            break

            elif t == p.QUEEN:
                for dir_i in range(8):
                    for dst in RAYS[src][dir_i]:
                        counts[dst] += 1
                        if board[dst] != p.NONE:
                            break

        return tuple(counts)
    
    def attack_maps(
        self,
        *,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Return (white_attack_counts, black_attack_counts).
        """
        return (
            self.attack_counts(by_white=True, piece_types=piece_types),
            self.attack_counts(by_white=False, piece_types=piece_types),
        )
        
    def defense_count(
        self,
        square: int,
        *,
        white_piece: bool,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> int:
        """
        Count same-color defenders of a square occupied by a piece of `white_piece`.
        """
        return self.count_attackers(square, by_white=white_piece, piece_types=piece_types)


    def enemy_attack_count(
        self,
        square: int,
        *,
        white_piece: bool,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> int:
        """
        Count enemy attackers of a square occupied by a piece of `white_piece`.
        """
        return self.count_attackers(square, by_white=not white_piece, piece_types=piece_types)

    def piece_at(self, sq: int) -> int:
        return int(self._board[sq])

    def king_square(self, white: bool) -> int:
        return self._white_king_sq if white else self._black_king_sq


    def queen_square(self, white: bool) -> int | None:
        target = p.WHITE_QUEEN if white else p.BLACK_QUEEN

        for sq, pc in enumerate(self._board):
            if int(pc) == target:
                return sq

        return None
        
    def move_gives_check(self, move: Move) -> bool:
        side = self._is_white_to_move

        undo = self._do_move(move)
        try:
            return self.in_check(not side)
        finally:
            self._undo_move(undo)

    def move_gives_checkmate(self, move: Move) -> bool:
        side = self._is_white_to_move

        undo = self._do_move(move)
        try:
            done, reason = self.game_end_state()
            return done and reason == "checkmate" and self.in_check(not side)
        finally:
            self._undo_move(undo)

    def legal_moves_attacking_square(
        self,
        *,
        white: bool,
        square: int,
        piece_types: set[int] | frozenset[int] | None = None,
    ) -> list[Move]:
        """
        Return legal moves by `white` that result in `square` being attacked by `white`.

        Useful for tempo-threat features:
        - legal moves attacking the enemy queen
        - legal moves attacking loose pieces
        - legal moves attacking king-zone squares
        """
        old_turn = self._is_white_to_move
        old_ep = self._en_passant_target
        old_zkey = self._zkey

        try:
            self._is_white_to_move = white

            if white != old_turn:
                self._en_passant_target = None

            self._zkey = self._recompute_zobrist()

            moves = self.get_all_legal_moves()
            out: list[Move] = []

            for m in moves:
                moved_piece = int(self._board[m.src_square])
                if piece_types is not None and PTYPE[moved_piece] not in piece_types:
                    continue

                undo = self._do_move(m)
                try:
                    if self.is_square_attacked(square, by_white=white):
                        out.append(m)
                finally:
                    self._undo_move(undo)

            return out

        finally:
            self._is_white_to_move = old_turn
            self._en_passant_target = old_ep
            self._zkey = old_zkey

    @contextmanager
    def temporary_side_to_move(self, white: bool) -> Iterator[None]:
        """
        Temporarily set the side to move.

        Useful for static/evaluation queries where we want legal moves for either side.

        If querying the non-actual side to move, en-passant is disabled to avoid
        bogus EP moves.
        """
        old_turn = self._is_white_to_move
        old_ep = self._en_passant_target
        old_zkey = self._zkey

        try:
            self._is_white_to_move = white

            if white != old_turn:
                self._en_passant_target = None

            self._zkey = self._recompute_zobrist()

            yield

        finally:
            self._is_white_to_move = old_turn
            self._en_passant_target = old_ep
            self._zkey = old_zkey

def _on_board(sq: int) -> bool:
    return 0 <= sq < 64

def _same_file(a: int, b: int) -> bool:
    return (a & 7) == (b & 7)
