from __future__ import annotations
from typing import TYPE_CHECKING

import pygame as pg
import math

from gui.controller.controller import Controller

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.view.main_window import MainWindow

# Here will reside the main loop of the application

class Game:
    def __init__(self) -> None:
        self._ctrl: Controller = Controller(8,8)
        self._view: MainWindow = self._ctrl.get_view() # TODO: Uncouple view from game
    
    def run(self) -> None:
        last_print = pg.time.get_ticks()
        while self._ctrl.get_state() == Controller.RUNNING:
            
            for event in pg.event.get():
                self._ctrl.handle_event(event)
                    
            self._view.draw()
            
            now = pg.time.get_ticks()
            if now - last_print > 1000:
                print(f"fps = {self._view._clock.get_fps():.1f}")
                last_print = now
                    
        print("Exiting")
        pg.quit()
        
        
game = Game()
game.run()