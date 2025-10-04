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
        self._view: MainWindow = self._ctrl.get_view() # TODO: Uncouple view from game
    
    def run(self) -> None:

        while self._ctrl.get_state() == Controller.RUNNING:
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self._ctrl.update_state(Controller.STOPPED)
                    
                self._view.process_events(event)
                    
            self._view.draw()
                    
        print("Exiting")
        pg.quit()
        
        
game = Game()
game.run()