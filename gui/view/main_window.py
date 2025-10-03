from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg

from gui.view.sidebar import Sidebar
from gui.view.board_window import BoardWindow

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  

class MainWindow:
    
    def __init__(self, controller: Controller, width: int, height: int, title: str = "Chess") -> None:
        
        # Initialize Pygame
        pg.init()
        
        self._ctrl: Controller = controller
        
        self._widht:  int = width
        self._height: int = height
        self._tile:   str = title
        
        self._sidebar_width:  int = width - height
        self._sidebar_height: int = height
        
        self._board_width:  int = height
        self._board_height: int = height
        
        self._clock = pg.Clock()
        self._manager = pgg.UIManager((width, height))
        
        # Set up the game window
        self._screen = pg.display.set_mode((width, height))
        pg.display.set_caption(title)

        self._sidebar:   Sidebar = Sidebar(self._board_width, 0, width, height, self._manager)
        self._board: BoardWindow = BoardWindow(self._ctrl, 0, 0, 
                                               self._board_width, self._board_height, 
                                               self._manager)
        
        self._test_img = pg.image.load_sized_svg('assets/pieces/king-w.svg', (80,80)).convert_alpha()
        # self._test_img = pg.image.load('assets/Final.png').convert()
        
    def update_image(self, x: int, y: int) -> None:
        self._screen.fill((50, 50, 50))
        self._screen.blit(self._test_img, (x,y))
        
    def process_events(self, event):
        self._manager.process_events(event)
        
    def draw(self, dt: float):
        self._manager.update(dt)
        self._board.draw_chess_board()
        self._manager.draw_ui(self._screen)
        pg.display.flip()
                    
    def get_screen(self):
        return self._screen       

    def get_clock(self):
        return self._clock