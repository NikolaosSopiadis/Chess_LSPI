from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  
    
class BoardWindow:
    def __init__(self, controller: Controller, x0: int, y0: int,
                 width: int, height: int, manager) -> None:

        self._ctrl: Controller = controller
        
        self._widht:  int = width
        self._height: int = height
        
        self._manager = manager
        
        self._sidebard = pgg.elements.UIPanel 
        
        self._board_panel = pgg.elements.UIPanel(
            relative_rect = pg.Rect(x0, y0, width, height),
            starting_height = 0,
            manager = manager
        )
        
        self._board_surface = pg.Surface((width, height)).convert_alpha()

        self._board = pgg.elements.UIImage(
            relative_rect = pg.Rect(x0, y0, width, height),
            image_surface = self._board_surface,
            manager = manager,
            container = self._board_panel
        )
        
    def draw_chess_board(self) -> None:
        ranks: int = self._ctrl.get_ranks()
        files: int = self._ctrl.get_files()
        
        square_size: int = int(self._height / ranks)
        
        light_color: tuple[int, int, int] = (200, 180, 160)
        dark_color:  tuple[int, int, int] = (60, 50, 40)

        for f in range(files):
            for r in range(ranks):
                square = pg.Rect(f * square_size, r * square_size, square_size, square_size)
                color: tuple[int, int, int] = light_color if (r + f) % 2 == 0 else dark_color

                pg.draw.rect(self._board_surface, color, square)
                    
        # redraw board surface, then push it into the UIImage
        self._board.set_image(self._board_surface)  # update the displayed image 
        