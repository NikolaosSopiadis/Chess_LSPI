import numpy as np
import numpy.typing as npt

from chess_core.piece import Piece as p
from chess_core.move import Move, Promotion, MoveFlag

# TODO:
#       - Create get_legal_moves that checks for pins and checks
#       - Add checkmate check and draw check (
#           - same position 3 times
#           - 50 move rule
#           - insufficient material
#           - stalemate)


class Board:

    NO_PIECE: int    = 0
    ENEMY_PIECE: int = -1
    OWN_PIECE: int   = 1
    
    WHITE_CASTLE_KINGSIDE: int  = 1<<0 
    WHITE_CASTLE_QUEENSIDE: int = 1<<1 
    BLACK_CASTLE_KINGSIDE: int  = 1<<2 
    BLACK_CASTLE_QUEENSIDE: int = 1<<3
    
    EN_PASSANT_CHECK_WHITE: int = 0
    EN_PASSANT_CHECK_BLACK: int = 1
    
    def __init__(self, ranks=8, files=8):
        self._ranks:     int = ranks
        self._files:     int = files
        self._grid_size: int = ranks * files

        self._board: npt.NDArray[np.uint8] = np.zeros(self._grid_size, dtype=np.uint8)
        
        self._castling_rights: int = ( self.WHITE_CASTLE_KINGSIDE 
                                     | self .WHITE_CASTLE_QUEENSIDE
                                     | self.BLACK_CASTLE_KINGSIDE 
                                     | self.BLACK_CASTLE_QUEENSIDE)

        self._en_passant_target: int | None = None
        
        self._is_white_to_move: bool = True

        self._init_board()
        
    def _init_board(self) -> None:
        self._board: npt.NDArray[np.uint8] = np.zeros(self._grid_size, dtype=np.uint8)
        self._board[0] = p.WHITE_ROOK
        self._board[1] = p.WHITE_KNIGHT
        self._board[2] = p.WHITE_BISHOP
        self._board[3] = p.WHITE_QUEEN
        self._board[4] = p.WHITE_KING
        self._board[5] = p.WHITE_BISHOP 
        self._board[6] = p.WHITE_KNIGHT
        self._board[7] = p.WHITE_ROOK
        for i in range(8, 16):
            self._board[i] = p.WHITE_PAWN

        for i in range(48, 56):
            self._board[i] = p.BLACK_PAWN
        self._board[56] = p.BLACK_ROOK
        self._board[57] = p.BLACK_KNIGHT
        self._board[58] = p.BLACK_BISHOP
        self._board[59] = p.BLACK_QUEEN
        self._board[60] = p.BLACK_KING
        self._board[61] = p.BLACK_BISHOP 
        self._board[62] = p.BLACK_KNIGHT
        self._board[63] = p.BLACK_ROOK
        
    # TODO: cleanup make_move by better utilizing the flags in a match case block
    def make_move(self, move: Move) -> bool:
        src: int = move.src_square
        dst: int = move.dst_square

        # Out of bounds
        if src < 0 or src >= self._grid_size:
            return False
        
        if dst < 0 or dst >= self._grid_size:
            return False
        
        src_piece = self._board[src]
        dst_piece = self._board[dst]
        
        # Ignore empty squares
        if src_piece == p.NONE:
            return False
        
        # Ignore moves from and to the same square
        if src == dst:
            return False

        f_src, r_src = self.idx_to_f_r(src)
        f_dst, r_dst = self.idx_to_f_r(dst)
        src_idx: int = self.get_idx(f_src, r_src)

        # Only make legal moves
        legal_moves: list[Move] = self.get_legal_moves(src_idx)       
        # Currently move has no flags so it only moves to empty squares will match
        # if move not in legal_moves:
        if all(m.dst_square != dst for m in legal_moves):
            return False

        # Reset en passant
        # en_passant_prev: int | None = self._en_passant_target
        self._en_passant_target = None

        match p.piece_type(src_piece):
            case p.PAWN:
                # Update en passant on double pawn move
                if move.check_flag(MoveFlag.DOUBLE_PAWN):
                # if r_dst - r_src == 2 or r_dst - r_src == -2:
                    self._en_passant_target = self.get_idx(f_dst, (r_src + r_dst)//2)
                elif move.check_flag(MoveFlag.PROMOTION):
                    promo_piece: int
                    piece_color: int = p.piece_color(src_piece)
                    match move.promotion:
                        case Promotion.QUEEN:
                            promo_piece = p.make_piece(p.QUEEN, piece_color)
                        case Promotion.KNIGHT:
                            promo_piece = p.make_piece(p.KNIGHT, piece_color)
                        case Promotion.BISHOP:
                            promo_piece = p.make_piece(p.BISHOP, piece_color)
                        case Promotion.ROOK:
                            promo_piece = p.make_piece(p.ROOK, piece_color)
                        case _:
                            raise AssertionError("Promotion flag set with no promotion piece selected")

                    self._board[src] = promo_piece
                elif move.check_flag(MoveFlag.EN_PASSANT):
                # if en_passant_prev == dst:
                    if self._is_white_to_move: 
                        self._board[dst - self._files] = p.NONE 
                    else:
                        self._board[dst + self._files] = p.NONE 
                    

            case p.KING:
                qs_dst = self.get_idx(2, r_src)             # c-file
                ks_dst = self.get_idx(self._files-2, r_src) # g-file
                
                # Update castling rights
                # Don't check for obstructions as they are already checked in get_legal_moves
                if self._is_white_to_move:
                    
                    if self._has_castling_rights(self.WHITE_CASTLE_QUEENSIDE) and dst == qs_dst:
                        # Castle queenside
                        # move rook a-file to d-file
                        self._board[self.get_idx(3, r_src)] = p.WHITE_ROOK 
                        self._board[self.get_idx(0, r_src)] = p.NONE
                        
                    elif self._has_castling_rights(self.WHITE_CASTLE_KINGSIDE) == True and dst == ks_dst:
                        # Castle kingside
                        # move rook h-file to f-file
                        self._board[self.get_idx(self._files-3, r_src)] = p.WHITE_ROOK
                        self._board[self.get_idx(self._files-1, r_src)] = p.NONE

                    self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                    self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)
                else:
                    
                    if self._has_castling_rights(self.BLACK_CASTLE_QUEENSIDE) and dst == qs_dst:
                        # Castle queenside
                        # move rook a-file to d-file
                        self._board[self.get_idx(3, r_src)] = p.BLACK_ROOK 
                        self._board[self.get_idx(0, r_src)] = p.NONE

                    elif self._has_castling_rights(self.BLACK_CASTLE_KINGSIDE) and dst == ks_dst:
                        # Castle kingside
                        # move rook h-file to f-file
                        self._board[self.get_idx(self._files-3, r_src)] = p.BLACK_ROOK
                        self._board[self.get_idx(self._files-1, r_src)] = p.NONE

                    self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                    self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)

                # TODO: change castling to use its flag
                # if(move.check_flag(MoveFlag.CASTLE)):
                    

            case p.ROOK:
                # Update castling rights
                # Left 
                if f_src == 0: 
                    if self._is_white_to_move:
                        self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                    else:
                        self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                # Right
                elif f_src == self._files - 1:
                    if self._is_white_to_move:
                        self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)
                    else:
                        self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)

        match p.piece_type(dst_piece):
            case p.ROOK:
                # Remove castling rights if rook is captured
                # White captured black's rook
                if self._is_white_to_move:
                    if dst == self._grid_size - self._files - 1:
                        self._clear_castling_rights(self.BLACK_CASTLE_QUEENSIDE)
                    elif dst == self._grid_size - 1:
                        self._clear_castling_rights(self.BLACK_CASTLE_KINGSIDE)
                # Black captured white's rook
                else:
                    if dst == 0:
                        self._clear_castling_rights(self.WHITE_CASTLE_QUEENSIDE)
                    elif dst == self._files - 1:
                        self._clear_castling_rights(self.WHITE_CASTLE_KINGSIDE)

        # # Captured en passant
        # if move.check_flag(MoveFlag.EN_PASSANT):
        # # if en_passant_prev == dst:
        #     if self._is_white_to_move: 
        #         self._board[dst - self._files] = p.NONE 
        #     else:
        #         self._board[dst + self._files] = p.NONE 

        self._board[dst] = self._board[src]
        self._board[src] = p.NONE 
        
        # Change turn
        self._is_white_to_move = not self._is_white_to_move
        
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

    # Returns 0 on empty, -1 on enemy piece and 1 on own piece
    def _check_enemy_piece(self, dst: int) -> int:
        piece_at_dst: int = self._board[dst]
        if piece_at_dst == p.NONE:
            return self.NO_PIECE

        # White to move
        if self._is_white_to_move: 
            if p.is_white(piece_at_dst):
                return self.OWN_PIECE
            return self.ENEMY_PIECE
        
        # Black to move
        if p.is_white(piece_at_dst):
            return self.ENEMY_PIECE
        return self.OWN_PIECE

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
        
        if self._has_castling_rights(self.WHITE_CASTLE_QUEENSIDE if white else self.BLACK_CASTLE_QUEENSIDE):
            obstructed = False
            for f in range(f_src - 1, 0, -1):
                castle_path = self.get_idx(f, r_src)
                if self._check_enemy_piece(castle_path) != self.NO_PIECE:
                    obstructed = True
                    break
            if not obstructed:
                moves.append(Move.castle(src, queenside_dst))

        if self._has_castling_rights(self.WHITE_CASTLE_KINGSIDE if white else self.BLACK_CASTLE_KINGSIDE):
            obstructed = False
            for f in range(f_src + 1, self._files - 1):
                castle_path = self.get_idx(f, r_src)
                if self._check_enemy_piece(castle_path) != self.NO_PIECE:
                    obstructed = True
                    break
            if not obstructed:
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
        pseudo: list[Move] = self.get_pseudolegal_moves(src)
        legal: list[Move] = []
        piece: int = self._board[src]
        white_to_move: bool = p.is_white(piece)

        king_sq: int = self._find_king(white_to_move)
        for move in pseudo:
            # make a temporary copy
            board_copy: npt.NDArray[np.uint8] = self._board.copy()
            board_copy[move.dst_square] = board_copy[move.src_square]
            board_copy[move.src_square] = p.NONE

            # find new king position if moved
            king_pos: int = move.dst_square if p.piece_type(piece) == p.KING else king_sq

            if not self.is_square_attacked(king_pos, by_white=not white_to_move):
                legal.append(move)

        return legal


    def get_board(self) -> npt.NDArray[np.uint8]:
        return self._board

    def _push_move(self, moves: list[Move], src: int, dst: int,
                   promotion: Promotion = Promotion.NONE,
                   en_passant: bool = False, castle: bool = False,
                   double_pawn: bool = False) -> None:
        captured: int = self._board[dst] # For en passant, calculate capture in make_move

        if castle:
            moves.append(Move.castle(src, dst))
        elif en_passant:
            moves.append(Move.en_passant(src, dst))
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
        board: npt.NDArray[np.uint8] = self._board

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
        king_piece: int = p.WHITE_KING if white else p.BLACK_KING
        idxs = np.where(self._board == king_piece)[0]
        return int(idxs[0]) if len(idxs) > 0 else -1

