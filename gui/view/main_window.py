import pygame as pg

class MainWindow:
    
    def __init__(self, width, height, title="Chess") -> None:
        
        self._widht  = width
        self._height = height
        self._tile   = title
        
        # Initialize Pygame
        pg.init()
        
        # Set up the game window
        screen = pg.display.set_mode((width, height))
        pg.display.set_caption(title)
        