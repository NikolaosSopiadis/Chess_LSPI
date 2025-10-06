import numpy as np
import numpy.typing as npt
import pygame as pg
import pygame_gui as pgg

from gui.view.main_window import MainWindow
from chess_core.piece import Piece as p     
class Controller:
    
    STOPPED: int = 0
    RUNNING: int = 1
    
    def __init__(self) -> None:
        self._state: int = self.RUNNING
        self._ranks: int = 8
        self._files: int = 8
        
        self._view: MainWindow = MainWindow(self, 800*2, 600*2, "Chess")
        
        grid_size = self._ranks * self._files
        self._board: npt.NDArray[np.uint8] = np.zeros(grid_size, dtype=np.uint8)
        self._board[0] = p.WHITE_ROOK
        self._board[1] = p.WHITE_KNIGHT
        self._board[2] = p.WHITE_BISHOP
        self._board[3] = p.WHITE_QUEEN
        self._board[4] = p.WHITE_KING
        self._board[5] = p.WHITE_BISHOP 
        self._board[6] = p.WHITE_KNIGHT
        self._board[7] = p.WHITE_ROOK

        self._board[56] = p.BLACK_ROOK
        self._board[57] = p.BLACK_KNIGHT
        self._board[58] = p.BLACK_BISHOP
        self._board[59] = p.BLACK_QUEEN
        self._board[60] = p.BLACK_KING
        self._board[61] = p.BLACK_BISHOP 
        self._board[62] = p.BLACK_KNIGHT
        self._board[63] = p.BLACK_ROOK
        
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
        # board: npt.NDArray[np.uint8] = model.get_board
        return self._board
    
    def get_piece_sprite(self, piece:int) -> str:
        color: str = "w" if p.is_white(piece) else "b"
        folder_path: str = "assets/pieces/"

        match p.piece_type(piece):
            case p.NONE:
                return ""
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
                self._view.update_mouse_pos(event.pos)
                
            case pg.MOUSEBUTTONDOWN:
                self._view.select_square()
                
            case pg.MOUSEBUTTONUP:
                print("Mouse up")
                
            case pgg.UI_BUTTON_PRESSED:
                
                match event.ui_element:
                    case self._view._sidebar._test_button:
                        print("Button pressed")
                        self._board[20] = self._board[0]
                        self._board[0] = p.NONE
                        
                