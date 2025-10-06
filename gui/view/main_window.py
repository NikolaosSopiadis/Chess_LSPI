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
        
        self._width:  int = width
        self._height: int = height
        self._title:  str = title
        
        self._sidebar_width:  int = width - height
        self._sidebar_height: int = height
        
        self._board_width:  int = height
        self._board_height: int = height
        
        self._clock:        pg.Clock = pg.Clock()
        self._manager: pgg.UIManager = pgg.UIManager((width, height))

        self._max_fps: int = 144
        self._dt:      float = self._clock.tick(self._max_fps) / 1000
        
        # Set up the game window
        self._screen: pg.Surface = pg.display.set_mode((width, height), vsync=1)
        pg.display.set_caption(title)

        self._sidebar:   Sidebar = Sidebar(self._board_width, 0,
                                           self._sidebar_width, self._sidebar_height, 
                                           self._manager, self._ctrl)
        self._board: BoardWindow = BoardWindow(self._ctrl, 0, 0, 
                                               self._board_width, self._board_height, 
                                               self._manager)
        
    def _update_clock(self) -> None:
        self._dt = self._clock.tick(self._max_fps) / 1000
        self._dt = max(0.001, min(0.1, self._dt))
        
    def manager_process_events(self, event):
        self._manager.process_events(event)
        
    def draw(self):
        self._update_clock()
        self._manager.update(self._dt)
        self._screen.fill((50, 50, 50))

        self._board.draw()
        self._manager.draw_ui(self._screen)
        pg.display.flip()
        
    def update_mouse_pos(self, pos: tuple[float, float]) -> None:
        self._board.set_mouse_pos(pos)
        self._board.check_hover()
        
    def mouse_clicked(self, clicked: bool, mouse_pos: tuple[float, float]) -> None:
        self._board.set_mouse_clicked(clicked)
        
    def select_square(self):
        self._board.select_square()