from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg

from gui.view.sidebar import Sidebar
from gui.view.board_window import BoardWindow

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  

# TODO: stop using pgg ui_image for the board as it is very expensive.
class MainWindow:
    
    def __init__(self, controller: Controller, width: int, height: int, title: str = "Chess") -> None:
        
        # Initialize Pygame
        pg.init()
        
        self._ctrl: Controller = controller
        
        self._width:  int = width
        self._height: int = height
        self._title:  str = title
        
        self._board_width, self._board_height, self._sidebar_width, self._sidebar_height = (
            self._compute_layout(width, height)
        )
        
        self._clock:        pg.Clock = pg.Clock()
        self._manager: pgg.UIManager = pgg.UIManager((width, height), theme_path="assets/theme.json")

        self._max_fps: int = 60
        self._dt:      float = self._clock.tick(self._max_fps) / 1000
        
        # Set up the game window
        self._screen: pg.Surface = pg.display.set_mode((width, height), pg.RESIZABLE, vsync=1)
        pg.display.set_caption(title)

        self._sidebar:   Sidebar = Sidebar(self._board_width, 0,
                                           self._sidebar_width, self._sidebar_height, 
                                           self._manager, self._ctrl)
        self._board: BoardWindow = BoardWindow(self._ctrl, 0, 0, 
                                               self._board_width, self._board_height, 
                                               self._manager)
        
    def _update_clock(self) -> None:
        self._dt = self._clock.tick(self._max_fps) / 1000
        # self._dt = max(0.001, min(0.1, self._dt))
        
    def manager_process_events(self, event):
        self._manager.process_events(event)
        
    def draw(self):
        self._update_clock()
        self._manager.update(self._dt)
        self._screen.fill((50, 50, 50))

        self._board.draw()
        self._manager.draw_ui(self._screen)
        pg.display.flip()
        
    def on_resize(self, new_width: int, new_height: int) -> None:
        self._width = new_width
        self._height = new_height

        self._board_width, self._board_height, self._sidebar_width, self._sidebar_height = (
            self._compute_layout(new_width, new_height)
        )

        print(
            f"resize: window=({new_width},{new_height}) "
            f"board=({self._board_width},{self._board_height}) "
            f"sidebar=({self._sidebar_width},{self._sidebar_height})"
        )

        self._manager.set_window_resolution((new_width, new_height))

        self._board.resize(self._board_width, self._board_height)
        self._sidebar.resize(
            self._board_width,
            0,
            self._sidebar_width,
            self._sidebar_height,
        )

    def on_mouse_down(self, mouse_pos: tuple[float, float]) -> None:
        return self._board.on_mouse_down(mouse_pos)
        
    def on_mouse_move(self, mouse_pos: tuple[float, float]) -> None:
        return self._board.on_mouse_move(mouse_pos)

    def on_mouse_up(self, mouse_pos: tuple[float, float]) -> None:
        return self._board.on_mouse_up(mouse_pos)

    def _compute_layout(self, width: int, height: int) -> tuple[int, int, int, int]:
        min_sidebar = 320

        if width <= min_sidebar + 100:
            # Very small window: still reserve sidebar, shrink board heavily.
            board_size = max(100, width - min_sidebar)
        else:
            board_size = min(height, width - min_sidebar)

        sidebar_width = max(min_sidebar, width - board_size)
        return board_size, board_size, sidebar_width, height