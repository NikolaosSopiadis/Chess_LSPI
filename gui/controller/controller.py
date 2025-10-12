import numpy as np
import numpy.typing as npt
import pygame as pg
import pygame_gui as pgg

from gui.view.main_window import MainWindow
from chess_core.piece import Piece as p     
from chess_core.board import Board
from chess_core.move import Move
class Controller:
    
    STOPPED: int = 0
    RUNNING: int = 1
    
    def __init__(self, ranks: int = 8, files: int = 8) -> None:
        self._state: int = self.RUNNING
        self._ranks: int = ranks
        self._files: int = files
        
        self._view: MainWindow = MainWindow(self, 800*1, 600*1, "Chess")
        self._model: Board     = Board(ranks, files)
        
        self._grid_size = self._ranks * self._files

    def get_state(self) -> int:
        return self._state
    
    def update_state(self, state: int) -> None:
        self._state = state
        
    def get_ranks(self) -> int:
        return self._ranks
    
    def get_files(self) -> int:
        return self._files
    
    # TODO: Remove this and interact directly with controller
    def get_view(self) -> MainWindow:
        return self._view
    
    def get_pieces_on_board(self) -> npt.NDArray[np.uint8]:
        return self._model.get_board()
    
    def get_piece_sprite(self, piece:int) -> str | None:
        color: str = "w" if p.is_white(piece) else "b"
        folder_path: str = "assets/pieces/"

        match p.piece_type(piece):
            case p.NONE:
                return None
            case p.PAWN:
                piece_type: str = "pawn-" + color + ".svg"
            case p.KNIGHT:
                piece_type: str = "knight-" + color + ".svg"
            case p.BISHOP:
                piece_type: str = "bishop-" + color + ".svg"
            case p.ROOK:
                piece_type: str = "rook-" + color + ".svg"
            case p.QUEEN:
                piece_type: str = "queen-" + color + ".svg"
            case p.KING:
                piece_type: str = "king-" + color + ".svg"
            case _:
                raise FileNotFoundError(f"Could find corresponding asset for piece: {piece}")
        return folder_path + piece_type

    def handle_event(self, event: pg.event.Event) -> None:
        
        self._view.manager_process_events(event)
        
        match event.type:
            case pg.QUIT:
                self._state = self.STOPPED
            
            case pg.KEYDOWN:
                
                match event.key:
                    case pg.K_q:
                        self._state = self.STOPPED
            
            case pg.MOUSEMOTION:
                self._view.on_mouse_move(event.pos)
                
            case pg.MOUSEBUTTONDOWN:
                self._view.on_mouse_down(event.pos)
                
            case pg.MOUSEBUTTONUP:
                self._view.on_mouse_up(event.pos)
                
            case pgg.UI_BUTTON_PRESSED:
                
                match event.ui_element:
                    case self._view._sidebar._test_button:
                        pass
                    
            case pg.VIDEORESIZE:
                x, y = event.size
                self._view.on_resize(x,y)
                    
    def move_piece(self, source: int, destination: int) -> bool:
        """Attempt to move a piece from source square to destination square

        Args:
            source (int): source square index
            destination (int): destination square index

        Returns:
            bool: True if success, False if illegal move
        """
        move = Move(source, destination)
        return self._model.make_move(move)
    
    # TODO: SOS Cache legal moves since they are asked multiple times per second (each frame a square is highlighted)
    # def get_legal_moves(self, square: int) -> list[int]:
        # file, rank = self._model.idx_to_f_r(square)
        # return self._model.get_legal_moves(file, rank)
        return self._model.get_legal_moves(square)
    
    def get_moves(self, src: int, legal: bool = True) -> list[Move]:
        if legal:
            return self._model.get_legal_moves(src)
        else:
            return self._model.get_pseudolegal_moves(src)

    def get_move_dests(self, src: int, legal: bool = True) -> list[int]:
        return [m.dst_square for m in self.get_moves(src, legal=legal)]
    
    def is_white_to_move(self) -> bool:
        return self._model.get_is_white_to_move()

    def is_friendly_piece(self, piece: int) -> bool:
        return piece != p.NONE and (p.is_white(piece) == self.is_white_to_move())
