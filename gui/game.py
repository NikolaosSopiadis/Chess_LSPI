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
        self._ctrl: Controller = Controller()
        self._view: MainWindow = self._ctrl.get_view()
    
    def run(self) -> None:
        t = 0
        x = 0
        y = 0
        
        delta_t: float = 0.1

        clock = pg.time.Clock()

        while self._ctrl.get_state() == Controller.RUNNING:
            
            self._view.draw_chess_board()
            
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self._ctrl.update_state(Controller.STOPPED)
                    
            pg.display.flip()
            
            delta_t = clock.tick(144) / 1000
            delta_t = max(0.001, min(0.1, delta_t))
                    
        print("Exiting")
        pg.quit()
        
        
game = Game()
game.run()