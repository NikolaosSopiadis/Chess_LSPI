from typing import Sequence
from dataclasses import dataclass

from chess_core.piece import Piece as p
from chess_core.move import Move, Promotion, MoveFlag

# TODO:
#       - Zobrist hashing for position repetition detection
#       - Add checkmate check and draw check (
#           - same position 3 times
#           - 50 move rule
#           - insufficient material
#           - stalemate)

# Perforamance TODO:
#      - Stop calling is_square_attacked() twice for castling legality check

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
    rook_src: int = -1                # for castling
    rook_dst: int = -1

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

        self._init_board()
        
    def _init_board(self) -> None:
        self.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        
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
    
    def idx_to_f_r(self, idx: int) -> tuple[int, int]:
        r: int = idx // self._files
        f: int = idx % self._files
        return f, r

    def get_idx(self, f: int, r: int) -> int:
        return f + (r * self._files)

    def _gen_pawn_moves(self, src: int) -> list[Move]:
        moves: list[Move] = []
        piece: int        = self._board[src]
        white: bool       = p.is_white(piece)
        f_src, r_src      = self.idx_to_f_r(src)
        rank_direction    = 1 if white else -1
        promotion_rank    = self._ranks - 1 if white else 0
        start_rank        = 1 if white else self._ranks - 2
 
        # One forward
        r1: int = r_src + rank_direction
        if 0 <= r1 < self._ranks:
            dst = self.get_idx(f_src, r1)
            if self._board[dst] == p.NONE:
                if r1 == promotion_rank:
                    for to in (Promotion.KNIGHT, Promotion.BISHOP, Promotion.ROOK, Promotion.QUEEN):
                        self._push_move(moves, src, dst, promotion=to)
                else:
                    self._push_move(moves, src, dst)   

                # Two forward
                if r_src == start_rank:
                    mid: int = self.get_idx(f_src, r_src + rank_direction)
                    r2 = r_src + 2 * rank_direction
                    dst2 = self.get_idx(f_src, r2)
                    if self._board[mid] == p.NONE and self._board[dst2] == p.NONE:
                        self._push_move(moves, src, dst2, double_pawn=True)

        # Captures (left/right)
        for df in (-1, 1):
            f1: int = f_src + df
            r1: int = r_src + rank_direction
            
            if 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                dst: int = self.get_idx(f1, r1)
                target_piece: int = self._board[dst]
                if self._is_enemy(target_piece, white):
                    if r1 == promotion_rank:
                        for to in (Promotion.KNIGHT, Promotion.BISHOP, Promotion.ROOK, Promotion.QUEEN):
                            self._push_move(moves, src, dst, promotion=to)
                    else:
                        self._push_move(moves, src, dst)   

                # En passant
                if self._en_passant_target == dst:
                    self._push_move(moves, src, dst, en_passant=True)
        
        return moves

    def _gen_king_moves(self, src:int) -> list[Move]:
        moves: list[Move] = []
        piece: int        = self._board[src]
        white: bool       = p.is_white(piece)
        f_src, r_src      = self.idx_to_f_r(src)

        # Move 1 square in all
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                
                f = f_src + df
                r = r_src + dr
                if 0 <= f < self._files and 0 <= r < self._ranks:
                    dst = self.get_idx(f, r)
                    target_piece = self._board[dst]
                    if target_piece == p.NONE or self._is_enemy(target_piece, white):
                        self._push_move(moves, src, dst)

        # Castling (obstructions only, checks and pins will be calculated later)
        queenside_dst = self.get_idx(2, r_src)
        kingside_dst  = self.get_idx(self._files - 2, r_src)
        
        start_king_sq = self.get_idx(4, 0 if white else 7)
        if src != start_king_sq:
            return moves  # no castling unless king is on e1/e8

        rook_piece = p.WHITE_ROOK if white else p.BLACK_ROOK
        
        if self._has_castling_rights(self.WHITE_CASTLE_QUEENSIDE if white else self.BLACK_CASTLE_QUEENSIDE):
            obstructed = False
            for f in range(f_src - 1, 0, -1):
                castle_path = self.get_idx(f, r_src)
                if self._board[castle_path] != p.NONE:
                    obstructed = True
                    break
            # queenside rook must exist
            qs_rook_sq = self.get_idx(0, r_src)
            if self._board[qs_rook_sq] == rook_piece and not obstructed:
                moves.append(Move.castle(src, queenside_dst))

        if self._has_castling_rights(self.WHITE_CASTLE_KINGSIDE if white else self.BLACK_CASTLE_KINGSIDE):
            obstructed = False
            for f in range(f_src + 1, self._files - 1):
                castle_path = self.get_idx(f, r_src)
                if self._board[castle_path] != p.NONE:
                    obstructed = True
                    break
            # kingside rook must exist
            ks_rook_sq = self.get_idx(self._files - 1, r_src)
            if self._board[ks_rook_sq] == rook_piece and not obstructed:
                moves.append(Move.castle(src, kingside_dst))
        
        return moves

    def _is_enemy(self, piece: int, white_to_move: bool) -> bool:
        if piece == p.NONE:
            return False
        return p.is_white(piece) != white_to_move

    def _gen_knight_moves(self, src: int) -> list[Move]:
        moves: list[Move] = []
        piece: int        = self._board[src]
        white: bool       = p.is_white(piece)

        f0, r0 = self.idx_to_f_r(src)
        jumps  = ((-1,+2), (+1,+2), (-2,+1), (+2,+1),
                  (-2,-1), (+2,-1), (-1,-2), (+1,-2))

        for df, dr in jumps:
            f: int = f0 + df
            r: int = r0 + dr
            if 0 <= f < self._files and 0 <= r < self._ranks:
                dst: int = self.get_idx(f, r)
                target_piece: int = self._board[dst]
                if target_piece == p.NONE or self._is_enemy(target_piece, white):
                    self._push_move(moves, src, dst)

        return moves


    def _gen_ray(self, src: int, df: int, dr: int) -> list[Move]:
        """Scan in one direction until blocked; return quiet/capture moves."""
        moves: list[Move] = []
        piece: int        = self._board[src]
        white: bool       = p.is_white(piece)
        f, r              = self.idx_to_f_r(src)

        while True:
            f += df
            r += dr
            if not (0 <= f < self._files and 0 <= r < self._ranks):
                break

            dst: int    = self.get_idx(f, r)
            target_piece: int = self._board[dst]
            if target_piece == p.NONE:
                self._push_move(moves, src, dst)
                continue
            if self._is_enemy(target_piece, white):
                self._push_move(moves, src, dst)
            # hit something (enemy or own): stop the ray
            break

        return moves

    def _gen_diag_moves(self, src: int) -> list[Move]:
        # bishop-like
        moves: list[Move] = []
        moves += self._gen_ray(src, +1, +1)
        moves += self._gen_ray(src, +1, -1)
        moves += self._gen_ray(src, -1, +1)
        moves += self._gen_ray(src, -1, -1)
        return moves

    def _gen_ortho_moves(self, src: int) -> list[Move]:
        # rook-like
        moves: list[Move] = []
        moves += self._gen_ray(src, +1,  0)
        moves += self._gen_ray(src, -1,  0)
        moves += self._gen_ray(src,  0, +1)
        moves += self._gen_ray(src,  0, -1)
        return moves

    def _gen_bishop_moves(self, src: int) -> list[Move]:
        return self._gen_diag_moves(src)

    def _gen_rook_moves(self, src: int) -> list[Move]:
        return self._gen_ortho_moves(src)

    def _gen_queen_moves(self, src: int) -> list[Move]:
        return self._gen_diag_moves(src) + self._gen_ortho_moves(src)
    
    def get_pseudolegal_moves(self, src: int) -> list[Move]:
        piece: int = self._board[src]
        
        if piece == p.NONE or self._is_enemy(piece, self._is_white_to_move):
            return []
        
        match p.piece_type(piece):
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
            if m.check_flag(MoveFlag.CASTLE):
                if self.is_square_attacked(king_sq0, by_white=not side):
                    continue
                f_src, r_src = self.idx_to_f_r(m.src_square)
                f_dst, _     = self.idx_to_f_r(m.dst_square)
                if f_dst > f_src:  # kingside
                    through = [self.get_idx(f_src + 1, r_src), self.get_idx(f_src + 2, r_src)]
                else:              # queenside
                    through = [self.get_idx(f_src - 1, r_src), self.get_idx(f_src - 2, r_src)]
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

    def _push_move(self, moves: list[Move], src: int, dst: int,
                   promotion: Promotion = Promotion.NONE,
                   en_passant: bool = False, castle: bool = False,
                   double_pawn: bool = False) -> None:
        captured: int = self._board[dst] # For en passant, calculate capture in make_move

        if castle:
            moves.append(Move.castle(src, dst))
        elif en_passant:
            # captured pawn is behind dst
            src_piece = int(self._board[src])
            cap = p.BLACK_PAWN if p.is_white(src_piece) else p.WHITE_PAWN
            moves.append(Move(src, dst, MoveFlag.EN_PASSANT | MoveFlag.CAPTURE, Promotion.NONE, cap))
        elif promotion:
            moves.append(Move.promotion_to(src, dst, promotion, captured))
        elif double_pawn:
            moves.append(Move.double_pawn(src, dst))
        else:
            moves.append(Move.normal(src, dst, captured))

    def get_is_white_to_move(self) -> bool:
        return self._is_white_to_move
    
    def is_square_attacked(self, square: int, by_white: bool) -> bool:
        """Return True if `square` is attacked by the given color."""
        f_src, r_src = self.idx_to_f_r(square)
        board: Sequence[int] = self._board

        # --- Pawn attacks ---
        pawn_dir: int = -1 if by_white else +1
        for df in (-1, 1):
            f1: int = f_src + df
            r1: int = r_src + pawn_dir
            if 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                idx = self.get_idx(f1, r1)
                if board[idx] == (p.WHITE_PAWN if by_white else p.BLACK_PAWN):
                    return True

        # --- Knight attacks ---
        for df, dr in ((1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)):
            f1: int = f_src + df
            r1: int = r_src + dr
            if 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                idx = self.get_idx(f1, r1)
                if board[idx] == (p.WHITE_KNIGHT if by_white else p.BLACK_KNIGHT):
                    return True

        # --- Sliding pieces (Bishop / Rook / Queen) ---
        # Diagonals for Bishop/Queen
        for df, dr in ((1,1),(1,-1),(-1,1),(-1,-1)):
            f1: int = f_src + df
            r1: int = r_src + dr
            while 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                idx = self.get_idx(f1, r1)
                piece = board[idx]
                if piece == p.NONE:
                    f1 += df; r1 += dr
                    continue
                if p.is_white(piece) == by_white:
                    t = p.piece_type(piece)
                    if t in (p.BISHOP, p.QUEEN):
                        return True
                break  # blocked

        # Orthogonals for Rook/Queen
        for df, dr in ((1,0),(-1,0),(0,1),(0,-1)):
            f1: int = f_src + df
            r1: int = r_src + dr
            while 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                idx = self.get_idx(f1, r1)
                piece = board[idx]
                if piece == p.NONE:
                    f1 += df; r1 += dr
                    continue
                if p.is_white(piece) == by_white:
                    t = p.piece_type(piece)
                    if t in (p.ROOK, p.QUEEN):
                        return True
                break  # blocked

        # --- King attacks ---
        for df, dr in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            f1: int = f_src + df
            r1: int = r_src + dr
            if 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                idx = self.get_idx(f1, r1)
                if board[idx] == (p.WHITE_KING if by_white else p.BLACK_KING):
                    return True

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
        )

        # reset en passant by default
        self._en_passant_target = None

        # halfmove clock (for 50-move draw)
        # reset on pawn move or any capture
        if p.piece_type(moved_piece) == p.PAWN or move.check_flag(MoveFlag.CAPTURE):
            self._halfmove_clock = 0
        else:
            self._halfmove_clock += 1

        # --- special moves ---
        if move.check_flag(MoveFlag.EN_PASSANT):
            # capture pawn behind destination
            if p.is_white(moved_piece):
                cap_sq = dst - self._files
            else:
                cap_sq = dst + self._files
            undo.captured_square = cap_sq
            undo.captured_piece = int(self._board[cap_sq])
            self._board[cap_sq] = p.NONE

        if move.check_flag(MoveFlag.CASTLE):
            f_src, r_src = self.idx_to_f_r(src)
            f_dst, _ = self.idx_to_f_r(dst)
            # kingside: dst file is 6; queenside: dst file is 2
            if f_dst > f_src:
                rook_src = self.get_idx(self._files - 1, r_src)
                rook_dst = self.get_idx(self._files - 3, r_src)
            else:
                rook_src = self.get_idx(0, r_src)
                rook_dst = self.get_idx(3, r_src)
            undo.rook_src, undo.rook_dst = rook_src, rook_dst
            self._board[rook_dst] = self._board[rook_src]
            self._board[rook_src] = p.NONE

        # update castling rights due to king/rook moves or rook capture
        t = p.piece_type(moved_piece)
        if t == p.KING:
            if p.is_white(moved_piece):
                self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE | self.WHITE_CASTLE_QUEENSIDE)
            else:
                self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE | self.BLACK_CASTLE_QUEENSIDE)
        elif t == p.ROOK:
            f_src, r_src = self.idx_to_f_r(src)
            if p.is_white(moved_piece):
                if f_src == 0 and r_src == 0: self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                if f_src == 7 and r_src == 0: self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)
            else:
                if f_src == 0 and r_src == 7: self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                if f_src == 7 and r_src == 7: self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)

        if p.piece_type(undo.captured_piece) == p.ROOK:
            f_dst, r_dst = self.idx_to_f_r(dst)
            if p.is_white(undo.captured_piece):
                if f_dst == 0 and r_dst == 0: self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                if f_dst == 7 and r_dst == 0: self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)
            else:
                if f_dst == 0 and r_dst == 7: self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                if f_dst == 7 and r_dst == 7: self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)

        # double pawn sets en passant target
        if move.check_flag(MoveFlag.DOUBLE_PAWN):
            f_src, r_src = self.idx_to_f_r(src)
            _, r_dst = self.idx_to_f_r(dst)
            self._en_passant_target = self.get_idx(f_src, (r_src + r_dst) // 2)

        # apply main piece move
        self._board[dst] = self._board[src]
        self._board[src] = p.NONE
 
        # update king square       
        if p.piece_type(moved_piece) == p.KING:
            if p.is_white(moved_piece):
                self._white_king_sq = dst
            else:
                self._black_king_sq = dst

        # promotion replaces piece on dst
        if move.check_flag(MoveFlag.PROMOTION):
            color = p.piece_color(moved_piece)
            promo_t = {
                Promotion.QUEEN:  p.QUEEN,
                Promotion.ROOK:   p.ROOK,
                Promotion.BISHOP: p.BISHOP,
                Promotion.KNIGHT: p.KNIGHT,
            }.get(move.promotion)
            if promo_t is None:
                raise AssertionError("Promotion flag set but invalid promotion piece")
            self._board[dst] = p.make_piece(promo_t, color)

        # side to move flips
        self._is_white_to_move = not self._is_white_to_move
        return undo


    def _undo_move(self, undo: Undo) -> None:
        move = undo.move
        src, dst = move.src_square, move.dst_square

        # restore turn first (so helpers that depend on side can be sane)
        self._is_white_to_move = undo.prev_is_white_to_move

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
        if move.check_flag(MoveFlag.CASTLE):
            self._board[undo.rook_src] = self._board[undo.rook_dst]
            self._board[undo.rook_dst] = p.NONE
            
        # restore king square
        self._white_king_sq = undo.prev_white_king_sq
        self._black_king_sq = undo.prev_black_king_sq
        

    def get_all_legal_moves(self) -> list[Move]:
        legal: list[Move] = []
        side = self._is_white_to_move
        for src in range(self._grid_size):
            piece = int(self._board[src])
            if piece == p.NONE:
                continue
            if p.is_white(piece) != side:
                continue
            legal.extend(self.get_legal_moves(src))
        return legal

    def in_check(self, white: bool) -> bool:
        k = self._white_king_sq if white else self._black_king_sq
        return k != -1 and self.is_square_attacked(k, by_white=not white)

    def game_end_state(self) -> tuple[bool, str]:
        """(done, reason) where reason in {'checkmate','stalemate','playing'} for now."""
        moves = self.get_all_legal_moves()
        if moves:
            return False, "playing"
        side = self._is_white_to_move
        if self.in_check(side):
            return True, "checkmate"
        return True, "stalemate"


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
        return self.get_idx(f, r)

    def idx_to_algebraic(self, idx: int) -> str:
        f, r = self.idx_to_f_r(idx)
        return f"{chr(ord('a') + f)}{r + 1}"

    def set_fen(self, fen: str) -> None:
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Bad FEN: {fen}")

        placement, active, castling, ep = parts[:4]
        halfmove = int(parts[4]) if len(parts) >= 5 else 0
        fullmove = int(parts[5]) if len(parts) >= 6 else 1  # optional (but nice to keep)

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
                idx = self.get_idx(f, r)
                piece = self._FEN_TO_PIECE[ch]
                self._board[idx] = piece
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
                idx = self.get_idx(f, r)
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
