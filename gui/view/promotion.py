from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  

class PromotionDialog:
    
    def __init__(self, x0: int, y0: int, square_size: int, 
                 manager: pgg.UIManager, controller: Controller) -> None:
        
        self._x0:     int = x0
        self._y0:     int = y0
        self._width:  int = square_size
        self._height: int = 4 * square_size
        
        self._ctrl: Controller = controller
        
        self._promotion_rect: pg.Rect = pg.Rect(0 ,0, self._width, self._height)
        self._promotion_surface: pg.Surface = pg.Surface((self._width, self._height), pg.SRCALPHA)