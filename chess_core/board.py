import numpy as np
import numpy.typing as npt

from chess_core.piece import Piece as p
from chess_core.move import Move

class Board:

    NO_PIECE: int    = 0
    ENEMY_PIECE: int = -1
    OWN_PIECE: int   = 1
    
    CASTLE_QUEENSIDE: int = 0
    CASTLE_KINGSIDE: int  = 1
    CASTLE_WHITE: int     = 0
    CASTLE_BLACK: int     = 2
    
    WHITE_CASTLE_QUEENSIDE: int = CASTLE_WHITE | CASTLE_QUEENSIDE
    WHITE_CASTLE_KINGSIDE: int  = CASTLE_WHITE | CASTLE_KINGSIDE
    BLACK_CASTLE_QUEENSIDE: int = CASTLE_BLACK | CASTLE_QUEENSIDE
    BLACK_CASTLE_KINGSIDE: int  = CASTLE_BLACK | CASTLE_KINGSIDE
    
    EN_PASSANT_CHECK_WHITE: int = 0
    EN_PASSANT_CHECK_BLACK: int = 1
    
    def __init__(self, ranks=8, files=8):
        self._ranks:     int = ranks
        self._files:     int = files
        self._grid_size: int = ranks * files

        self._board: npt.NDArray[np.uint8] = np.zeros(self._grid_size, dtype=np.uint8)
        
        # 0x: white, 1x:black
        # x0: left, x1: right
        self._can_castle: list[bool] = [True, True, True, True] 
        # self._can_pawn_be_taken_with_en_passant[color][file]
        self._can_pawn_be_taken_with_en_passant: npt.NDArray[np.bool] = np.full((2,self._files), False)
        
        self._is_white_to_move: bool = True

        self._init_board()
        # self._board[0] = p.WHITE_ROOK
        
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

        # Reset en passant board
        col: int = self.EN_PASSANT_CHECK_WHITE if self._is_white_to_move else self.EN_PASSANT_CHECK_BLACK
        self._can_pawn_be_taken_with_en_passant[col] = np.full(self._files, False)

        match p.piece_type(src_piece):
            case p.PAWN:
                # Update en passant
                if r_dst - r_src == 2 or r_dst - r_src == -2:
                    self._can_pawn_be_taken_with_en_passant[col][f_dst] = True

            case p.KING:
                # Update castling rights
                # Don't check for obstructions as they are already checked in get_legal_moves
                if self._is_white_to_move:
                    
                    if self._can_castle[self.WHITE_CASTLE_QUEENSIDE] == True and dst == 1:
                        # Castle queenside
                        self._board[dst + 1] = p.WHITE_ROOK
                        self._board[dst - 1] = p.NONE
                    elif self._can_castle[self.WHITE_CASTLE_KINGSIDE] == True and dst == self._files - 2:
                        # Castle kingside
                        self._board[dst - 1] = p.WHITE_ROOK
                        self._board[dst + 1] = p.NONE

                    self._can_castle[self.WHITE_CASTLE_QUEENSIDE] = False
                    self._can_castle[self.WHITE_CASTLE_KINGSIDE]  = False
                else:
                    
                    if self._can_castle[self.BLACK_CASTLE_QUEENSIDE] == True and dst == self._grid_size - 2:
                        # Castle queenside
                        self._board[dst - 1] = p.BLACK_ROOK
                        self._board[dst + 1] = p.NONE
                    elif self._can_castle[self.BLACK_CASTLE_KINGSIDE]  == True and dst == self._grid_size - self._files:
                        # Castle kingside
                        self._board[dst + 1] = p.BLACK_ROOK
                        self._board[dst - 1] = p.NONE

                    self._can_castle[self.BLACK_CASTLE_QUEENSIDE] = False
                    self._can_castle[self.BLACK_CASTLE_KINGSIDE]  = False

            case p.ROOK:
                # Update castling rights
                # Left 
                if f_src == 0: 
                    if self._is_white_to_move:
                        self._can_castle[self.WHITE_CASTLE_QUEENSIDE] = False
                    else:
                        self._can_castle[self.BLACK_CASTLE_QUEENSIDE] = False

                elif f_src == self._files - 1:
                    if self._is_white_to_move:
                        self._can_castle[self.WHITE_CASTLE_KINGSIDE] = False
                    else:
                        self._can_castle[self.BLACK_CASTLE_KINGSIDE] = False
                

        # Captured en passant
        enemy_col: int = self.EN_PASSANT_CHECK_BLACK if self._is_white_to_move else self.EN_PASSANT_CHECK_WHITE
        if self._can_pawn_be_taken_with_en_passant[enemy_col][f_dst] == True and \
            self._check_enemy_piece(dst) == self.NO_PIECE: 
        
                if self._is_white_to_move and \
                   self._check_enemy_piece(dst - self._files) == self.ENEMY_PIECE:                      
                        self._board[dst - self._files] = p.NONE 

                elif not self._is_white_to_move and \
                     self._check_enemy_piece(dst + self._files) == self.ENEMY_PIECE:                      
                        self._board[dst + self._files] = p.NONE 

        self._board[dst] = self._board[src]
        self._board[src] = p.NONE 
        
        # Change turn
        self._is_white_to_move = not self._is_white_to_move
        
        return True
    
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
                if self._check_enemy_piece(move_idx) == self.NO_PIECE:
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
                    if self._can_pawn_be_taken_with_en_passant[self.EN_PASSANT_CHECK_BLACK][f] == True and \
                       self._check_enemy_piece(move_idx) == self.NO_PIECE and \
                       self._check_enemy_piece(move_idx - self._files) == self.ENEMY_PIECE:                      
                            legal_moves.append(move_idx)

                # Capture left
                f = f_src - 1
                r = r_src + 1
                if f  >= 0:
                    move_idx = self.get_idx(f, r)
                    if self._check_enemy_piece(move_idx) == self.ENEMY_PIECE:
                        legal_moves.append(move_idx)
                        
                    # En passant
                    if self._can_pawn_be_taken_with_en_passant[self.EN_PASSANT_CHECK_BLACK][f] == True and \
                       self._check_enemy_piece(move_idx) == self.NO_PIECE and \
                       self._check_enemy_piece(move_idx - self._files) == self.ENEMY_PIECE:
                            legal_moves.append(move_idx)
        
        # Black pawn
        else:
            # If in starting rank, then can move two squares up
            if r_src == self._ranks - 2:
                f = f_src
                r = r_src - 2
                move_idx = self.get_idx(f, r)
                if self._check_enemy_piece(move_idx) == self.NO_PIECE:
                    legal_moves.append(move_idx)
                    
            # One move up
            f = f_src
            r = r_src - 1
            if r < self._ranks:
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
                    if self._can_pawn_be_taken_with_en_passant[self.EN_PASSANT_CHECK_WHITE][f] == True and \
                       self._check_enemy_piece(move_idx) == self.NO_PIECE and \
                       self._check_enemy_piece(move_idx + self._files) == self.ENEMY_PIECE:
                            legal_moves.append(move_idx)

                # Capture left
                f = f_src - 1
                r = r_src - 1
                if f  >= 0:
                    move_idx = self.get_idx(f, r)
                    if self._check_enemy_piece(move_idx) == self.ENEMY_PIECE:
                        legal_moves.append(move_idx)
                        
                    # En passant
                    if self._can_pawn_be_taken_with_en_passant[self.EN_PASSANT_CHECK_WHITE][f] == True and \
                       self._check_enemy_piece(move_idx) == self.NO_PIECE and \
                       self._check_enemy_piece(move_idx + self._files) == self.ENEMY_PIECE:
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

        if  r_src > 0 and f_src > 0:
            move_idx = self.get_idx(f_src - 1, r_src - 2)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 0 and f_src < self._files - 1:
            move_idx = self.get_idx(f_src + 1, r_src - 2)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 1 and f_src > 1:
            move_idx = self.get_idx(f_src - 2, r_src - 1)
            if self._check_enemy_piece(move_idx) != self.OWN_PIECE:
                legal_moves.append(move_idx)

        if  r_src > 1 and f_src < self._files - 2:
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
            if self._can_castle[self.WHITE_CASTLE_QUEENSIDE]:
                # Check for obstructions
                obstructed = False
                for i in range(1, self._files - f_src):
                    f = f_src - i
                    move_idx = self.get_idx(f, r_src)
                    if self._check_enemy_piece(move_idx) != self.NO_PIECE:
                        obstructed = True
                        break
                
                if not obstructed:
                    move_idx = self.get_idx(1, r_src)
                    legal_moves.append(move_idx)

            # Castle kingside
            if self._can_castle[self.WHITE_CASTLE_KINGSIDE]:
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
            if self._can_castle[self.BLACK_CASTLE_QUEENSIDE]:
                # Check for obstructions
                obstructed = False
                for i in range(1, self._files - f_src):
                    f = f_src - i
                    move_idx = self.get_idx(f, r_src)
                    if self._check_enemy_piece(move_idx) != self.NO_PIECE:
                        obstructed = True
                        break
                
                if not obstructed:
                    move_idx = self.get_idx(1, r_src)
                    legal_moves.append(move_idx)

            # Castle kingside
            if self._can_castle[self.BLACK_CASTLE_KINGSIDE]:
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
