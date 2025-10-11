import numpy as np
import numpy.typing as npt

from chess_core.piece import Piece as p
from chess_core.move import Move, Promotion, MoveFlag

# TODO: - Add promotions to make_move 
#       - Update get_[piece]_legal_moves for the rest of the pieces,
#       - Add promotion UI
#       - Maybe rename get_legal_moves to get_pseudolegal_moves
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

        # Only make legal moves
        legal_moves: list[int] = self.get_legal_moves(f_src, r_src)       
        if dst not in legal_moves:
            return False

        # Reset en passant
        en_passant_prev: int | None = self._en_passant_target
        self._en_passant_target = None

        match p.piece_type(src_piece):
            case p.PAWN:
                # Update en passant on double pawn move
                if r_dst - r_src == 2 or r_dst - r_src == -2:
                    self._en_passant_target = self.get_idx(f_dst, (r_src + r_dst)//2)

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

        # Captured en passant
        if en_passant_prev == dst:
            if self._is_white_to_move: 
                self._board[dst - self._files] = p.NONE 
            else:
                self._board[dst + self._files] = p.NONE 

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
        assert p.is_white(self._board[src]) == self._is_white_to_move

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

        # Capturesa (left/right)
        for df in (-1, 1):
            f1: int = r_src + df
            r1: int = r_src + rank_direction
            
            if 0 <= f1 < self._files and 0 <= r1 < self._ranks:
                dst: int = self.get_idx(f1, r1)
                target_piece: int = self._board[dst]
                if target_piece != p.NONE and (p.is_white(target_piece) != white):
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
        assert p.is_white(self._board[src]) == self._is_white_to_move

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
                    target = self._board[dst]
                    if target == p.NONE or (p.is_white(target) != white):
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

    def _get_pawn_legal_moves(self, src_square: int) -> list[int]:
        legal_moves: list[int] = list()
        src_piece: int = self._board[src_square]
     
        # Check if this is current players piece   
        if p.is_white(src_piece) and not self._is_white_to_move:
            return legal_moves
        
        if not p.is_white(src_piece) and self._is_white_to_move:
            return legal_moves
            
        move_idx: int
        f_src, r_src = self.idx_to_f_r(src_square)
        f: int
        r: int
        
        # White pawn
        if p.is_white(src_piece):
            # If in starting rank, then can move two squares up
            if r_src == 1:
                f = f_src
                r = r_src + 2
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) == self.NO_PIECE and \
                   self._check_enemy_piece(self.get_idx(f, r - 1)) == self.NO_PIECE:
                     legal_moves.append(move_idx)
                    
            # One move up
            f = f_src
            r = r_src + 1
            if r < self._ranks:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) == self.NO_PIECE:
                    legal_moves.append(move_idx)

                # Capture right
                f = f_src + 1
                r = r_src + 1
                if f < self._files:
                    move_idx = self.get_idx(f, r)
                    if self._check_enemy_piece(move_idx) == self.ENEMY_PIECE:
                        legal_moves.append(move_idx)
                        
                    # En passant
                    if self._en_passant_target == move_idx:
                        legal_moves.append(move_idx)

                # Capture left
                f = f_src - 1
                r = r_src + 1
                if f  >= 0:
                    move_idx = self.get_idx(f, r)
                    if self._check_enemy_piece(move_idx) == self.ENEMY_PIECE:
                        legal_moves.append(move_idx)
                        
                    # En passant
                    if self._en_passant_target == move_idx:
                        legal_moves.append(move_idx)
                    
        # Black pawn
        else:
            # If in starting rank, then can move two squares up
            if r_src == self._ranks - 2:
                f = f_src
                r = r_src - 2
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) == self.NO_PIECE and \
                   self._check_enemy_piece(self.get_idx(f, r + 1)) == self.NO_PIECE:
                     legal_moves.append(move_idx)
                    
            # One move up
            f = f_src
            r = r_src - 1
            if r >= 0:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) == self.NO_PIECE:
                    legal_moves.append(move_idx)

                # Capture right
                f = f_src + 1
                r = r_src - 1
                if f < self._files:
                    move_idx = self.get_idx(f, r)
                    if self._check_enemy_piece(move_idx) == self.ENEMY_PIECE:
                        legal_moves.append(move_idx)
                        
                    # En passant
                    if self._en_passant_target == move_idx:
                        legal_moves.append(move_idx)
                    
                # Capture left
                f = f_src - 1
                r = r_src - 1
                if f  >= 0:
                    move_idx = self.get_idx(f, r)
                    if self._check_enemy_piece(move_idx) == self.ENEMY_PIECE:
                        legal_moves.append(move_idx)
                        
                    # En passant
                    if self._en_passant_target == move_idx:
                        legal_moves.append(move_idx)
                    
        return legal_moves


    def _get_knight_legal_moves(self, src_square: int) -> list[int]:
        legal_moves: list[int] = list()
        src_piece: int = self._board[src_square]
     
        # Check if this is current players piece   
        if p.is_white(src_piece) and not self._is_white_to_move:
            return legal_moves
        
        if not p.is_white(src_piece) and self._is_white_to_move:
            return legal_moves
            
        move_idx: int
        f_src, r_src = self.idx_to_f_r(src_square)
        
        if  r_src < self._ranks - 2 and f_src > 0:
            move_idx = self.get_idx(f_src - 1, r_src + 2)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src < self._ranks - 2 and f_src < self._files - 1:
            move_idx = self.get_idx(f_src + 1, r_src + 2)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src < self._ranks - 1 and f_src > 1:
            move_idx = self.get_idx(f_src - 2, r_src + 1)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src < self._ranks - 1 and f_src < self._files - 2:
            move_idx = self.get_idx(f_src + 2, r_src + 1)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 1 and f_src > 0:
            move_idx = self.get_idx(f_src - 1, r_src - 2)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 1 and f_src < self._files - 1:
            move_idx = self.get_idx(f_src + 1, r_src - 2)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 0 and f_src > 1:
            move_idx = self.get_idx(f_src - 2, r_src - 1)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 0 and f_src < self._files - 2:
            move_idx = self.get_idx(f_src + 2, r_src - 1)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)
            
        return legal_moves

    def _get_king_legal_moves(self, src_square: int) -> list[int]:
        legal_moves: list[int] = list()
        src_piece: int = self._board[src_square]
     
        # Check if this is current players piece   
        if p.is_white(src_piece) and not self._is_white_to_move:
            return legal_moves
        
        if not p.is_white(src_piece) and self._is_white_to_move:
            return legal_moves
            
        move_idx: int
        f_src, r_src = self.idx_to_f_r(src_square)

        r: int
        f: int = f_src - 1
        if f >= 0:
            r = r_src - 1
            if r >= 0:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                    legal_moves.append(move_idx)

            r = r_src
            move_idx = self.get_idx(f, r)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

            r = r_src + 1
            if r < self._ranks:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                    legal_moves.append(move_idx)
            
        f = f_src 
        if f >= 0:
            r = r_src - 1
            if r >= 0:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                    legal_moves.append(move_idx)

            r = r_src + 1
            if r < self._ranks:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                    legal_moves.append(move_idx)
            
        f = f_src + 1
        if f < self._files:
            r = r_src - 1
            if r >= 0:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                    legal_moves.append(move_idx)

            r = r_src
            move_idx = self.get_idx(f, r)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

            r = r_src + 1
            if r < self._ranks:
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                    legal_moves.append(move_idx)

        obstructed: bool
        if self._is_white_to_move:
            # Castle queenside
            if self._has_castling_rights(self.WHITE_CASTLE_QUEENSIDE):
                # Check for obstructions
                obstructed = False
                for i in range(1, self._files - f_src):
                    f = f_src - i
                    move_idx = self.get_idx(f, r_src)
                    if self._check_enemy_piece(move_idx) != self.NO_PIECE:
                        obstructed = True
                        break
                
                if not obstructed:
                    move_idx = self.get_idx(2, r_src)
                    legal_moves.append(move_idx)

            # Castle kingside
            if self._has_castling_rights(self.WHITE_CASTLE_KINGSIDE):
                # Check for obstructions
                obstructed = False
                for i in range(1, self._files - f_src - 1):
                    f = f_src + i
                    move_idx = self.get_idx(f, r_src)
                    if self._check_enemy_piece(move_idx) != self.NO_PIECE:
                        # print(f"White kingside castling obstructed by idx = {move_idx}")
                        obstructed = True
                        break
                
                if not obstructed:
                    move_idx = self.get_idx(self._files - 2, r_src)
                    legal_moves.append(move_idx)

        else:
            # Castle queenside
            if self._has_castling_rights(self.BLACK_CASTLE_QUEENSIDE):
                # Check for obstructions
                obstructed = False
                for i in range(1, self._files - f_src):
                    f = f_src - i
                    move_idx = self.get_idx(f, r_src)
                    if self._check_enemy_piece(move_idx) != self.NO_PIECE:
                        obstructed = True
                        break
                
                if not obstructed:
                    move_idx = self.get_idx(2, r_src)
                    legal_moves.append(move_idx)

            # Castle kingside
            if self._has_castling_rights(self.BLACK_CASTLE_KINGSIDE):
                # Check for obstructions
                obstructed = False
                for i in range(1, self._files - f_src - 1):
                    f = f_src + i
                    move_idx = self.get_idx(f, r_src)
                    if self._check_enemy_piece(move_idx) != self.NO_PIECE:
                        # print(f"White kingside castling obstructed by idx = {move_idx}")
                        obstructed = True
                        break
                
                if not obstructed:
                    move_idx = self.get_idx(self._files - 2, r_src)
                    legal_moves.append(move_idx)
            
        return legal_moves

    def _get_diagonal_legal_moves(self, src_square: int) -> list[int]:
        legal_moves: list[int] = list()
        src_piece: int = self._board[src_square]
     
        # Check if this is current players piece   
        if p.is_white(src_piece) and not self._is_white_to_move:
            return legal_moves
        
        if not p.is_white(src_piece) and self._is_white_to_move:
            return legal_moves
            
        move_idx: int
        f_src, r_src = self.idx_to_f_r(src_square)

        m: int = max(self._files, self._ranks)

        for i in range(1, m):
            f: int = f_src - i
            r: int = r_src - i
            if f < 0 or r < 0:
                break
            
            move_idx = self.get_idx(f, r)
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break

        for i in range(1, m):
            f: int = f_src + i
            r: int = r_src + i
            if f >= self._files or r >= self._ranks:
                break
            
            move_idx = self.get_idx(f, r)
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break
        
        for i in range(1, m):
            f: int = f_src + i
            r: int = r_src - i
            if f >= self._files or r < 0:
                break
            
            move_idx = self.get_idx(f, r)
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break

        for i in range(1, m):
            f: int = f_src - i
            r: int = r_src + i
            if f < 0 or r >= self._ranks:
                break
            
            move_idx = self.get_idx(f, r)
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break

        return legal_moves

    def _get_orthogonal_legal_moves(self, src_square: int) -> list[int]:
        legal_moves: list[int] = list()
        src_piece: int = self._board[src_square]
     
        # Check if this is current players piece   
        if p.is_white(src_piece) and not self._is_white_to_move:
            return legal_moves
        
        if not p.is_white(src_piece) and self._is_white_to_move:
            return legal_moves
            
        move_idx: int
        f_src, r_src = self.idx_to_f_r(src_square)
        for f in range(1, r_src + 1):
            move_idx = src_square - f * self._files
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break
            
        for f in range(1, self._ranks - r_src):
            move_idx = src_square + f * self._files
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break

        for r in range(1, f_src + 1):
            move_idx = src_square - r
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break
            
        for r in range(1, self._files - f_src):
            move_idx = src_square + r
            match self._check_enemy_piece(move_idx):
                case self.NO_PIECE:
                    legal_moves.append(move_idx)
                case self.OWN_PIECE:
                    break
                case self.ENEMY_PIECE:
                    legal_moves.append(move_idx)
                    break

        return legal_moves

    def get_legal_moves(self, file: int, rank: int) -> list[int]:
        idx: int = self.get_idx(file, rank)
        piece = self._board[idx]
        legal_moves: list[int]
        
        match p.piece_type(piece):
            case p.PAWN:
                legal_moves = self._get_pawn_legal_moves(idx)

            case p.KNIGHT:
                legal_moves = self._get_knight_legal_moves(idx)

            case p.BISHOP:
                legal_moves = self._get_diagonal_legal_moves(idx)
            
            case p.ROOK:
                # legal_moves = self._get_orthogonal_legal_moves(file, rank)
                legal_moves = self._get_orthogonal_legal_moves(idx)

            case p.QUEEN:
                legal_moves = self._get_diagonal_legal_moves(idx) + self._get_orthogonal_legal_moves(idx)

            case p.KING:
                legal_moves = self._get_king_legal_moves(idx)

            case _:
                legal_moves = list()
                

        return legal_moves

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
