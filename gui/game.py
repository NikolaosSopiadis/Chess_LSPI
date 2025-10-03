import pygame as pg

from gui.view.main_window import MainWindow
from gui.controller.controller import Controller

# Here will reside the main loop of the application

class Game:
    def __init__(self) -> None:
        self._view: MainWindow = MainWindow(800, 600, "Chess")
        self._ctrl: Controller = Controller()
    
    def run(self) -> None:
        while self._ctrl.get_state() == Controller.RUNNING:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self._ctrl.update_state(Controller.STOPPED)
        print("Exiting")
        pg.quit()
        
game = Game()
game.run()