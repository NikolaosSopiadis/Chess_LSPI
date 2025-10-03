from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  

class MainWindow:
    
    def __init__(self, controller: Controller, width: int, height: int, title: str = "Chess") -> None:
        
        self._ctrl: Controller = controller
        
        self._widht:  int = width
        self._height: int = height
        self._tile:   str = title
        
        # Initialize Pygame
        pg.init()
        
        # Set up the game window
        self._screen = pg.display.set_mode((width, height))
        pg.display.set_caption(title)
        
        self._test_img = pg.image.load_sized_svg('assets/pieces/king-w.svg', (80,80)).convert_alpha()
        # self._test_img = pg.image.load('assets/Final.png').convert()
        
    def update_image(self, x: int, y: int) -> None:
        self._screen.fill((50, 50, 50))
        self._screen.blit(self._test_img, (x,y))
        
    def draw_chess_board(self) -> None:
        ranks: int = self._ctrl.get_ranks()
        files: int = self._ctrl.get_files()
        
        square_size: int = int(self._height / ranks)
        
        light_color: tuple[int, int, int] = (200, 180, 160)
        dark_color:  tuple[int, int, int] = (60, 50, 40)

        for f in range(files):
            for r in range(ranks):
                square = pg.Rect(f * square_size, r * square_size, square_size, square_size)
                if (r + f) % 2 == 0: # Light squares
                    pg.draw.rect(self._screen, light_color, square)
                else: # Dark squares
                    pg.draw.rect(self._screen, dark_color, square)
                    
    def get_screen(self):
        return self._screen       