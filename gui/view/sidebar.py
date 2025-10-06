from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  

class Sidebar:
    
    def __init__(self, x0: int, y0: int, width: int, height: int, manager: pgg.UIManager) -> None:
        self._x0:     int = x0
        self._y0:     int = 0
        self._width:  int = width
        self._height: int = height
        
        self._sidebar_rect: pg.Rect = pg.Rect(x0, self._y0, width, height)

        self._sidebar: pgg.elements.UIPanel = pgg.elements.UIPanel(
            relative_rect   = self._sidebar_rect,
            starting_height = self._y0,
            manager         = manager
        )
        
        self._title: pgg.elements.UILabel = pgg.elements.UILabel(
            relative_rect = pg.Rect(16, 10, width - 32, 28),
            text          = "Game Controls",
            manager       = manager,
            container     = self._sidebar
        )