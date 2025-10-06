from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg
import numpy as np
import numpy.typing as npt

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  
    
class BoardWindow:
    def __init__(self, controller: Controller, x0: int, y0: int,
                 width: int, height: int, manager) -> None:

        self._ctrl: Controller = controller
        self._manager = manager
        
        self._widht:         int = width
        self._height:        int = height
        self._board_flipped: bool = False
        
        self.flip_board()
        
        self._ranks: int = self._ctrl.get_ranks()
        self._files: int = self._ctrl.get_files()
        self._square_size: int = self._height // max(self._files, self._ranks)
        
        
        self._hovers_over_square: int = -1 # -1 if none else idx
        
        self._board_rect: pg.Rect = pg.Rect(x0, y0, width, height)
        
        # Might not be needed
        # self._board_panel: pgg.elements.UIPanel = pgg.elements.UIPanel(
        #     relative_rect = self._board_rect,
        #     starting_height = 0,
        #     manager = manager
        # )
        
        self._board_surface: pg.Surface = pg.Surface((width, height)).convert_alpha()

        self._board: pgg.elements.UIImage = pgg.elements.UIImage(
            relative_rect = self._board_rect,
            image_surface = self._board_surface,
            manager = manager,
            # container = self._board_panel
        )
        
    def draw_chess_board(self) -> None:
        light_color: tuple[int, int, int] = (200, 180, 160)
        dark_color:  tuple[int, int, int] = (60, 50, 40)
        
        pieces: npt.NDArray[np.uint8] = self._ctrl.get_pieces_on_board()

        for r in range(self._ranks):
            for f in range(self._files):
                square:             pg.Rect = pg.Rect(f * self._square_size, r * self._square_size, self._square_size, self._square_size)
                color: tuple[int, int, int] = light_color if (r + f) % 2 == 0 else dark_color
                pg.draw.rect(self._board_surface, color, square)
                
                piece_idx: int = self._get_piece_idx(r, f)

                font: pg.font.Font = pg.font.Font(None, 40)
                msg:           str = f"idx = {piece_idx}\nx,y = ({f},{r})\n\npiece = {pieces[piece_idx]}"
                text:   pg.Surface = font.render(msg, True, (255, 255, 255))
                self._board_surface.blit(text, square)
                self._draw_piece(pieces[piece_idx], square, self._square_size)
                    
        # redraw board surface, then push it into the UIImage
        self._board.set_image(self._board_surface)  # update the displayed image 
        
    def flip_board(self) -> None:
        self._board_flipped = not self._board_flipped
        
    def _get_piece_idx(self, rank: int, file: int) -> int:
        if self._board_flipped:
            piece_idx: int = self._files - file - 1 + (rank * self._files)
        else:
            piece_idx: int = file + ((self._ranks - rank - 1) * self._files)
        
        return piece_idx
    
    def _draw_piece(self, piece: int, square: pg.Rect, square_size: int) -> None:
        sprite_path: str = self._ctrl.get_piece_sprite(piece)
        
        # No piece to draw
        if sprite_path == "":
            return
        
        sprite: pg.Surface = pg.image.load_sized_svg(sprite_path, (square_size, square_size)).convert_alpha()
        self._board_surface.blit(sprite, square)
        
    def check_hover(self, mouse_pos: tuple[float, float]) -> None:

        # if mouse is outside of board, then no piece is selected
        if not self._board_rect.collidepoint(mouse_pos):
            self._hovers_over_square = -1
            return 
        self._hovers_over_square = self._idx_from_mouse_pos(mouse_pos)
        print(self._hovers_over_square)
        
    def _idx_from_mouse_pos(self, mouse_pos: tuple[float, float]) -> int:
        x: int = int(mouse_pos[0] - self._board_rect.left)
        y: int = int(mouse_pos[1] - self._board_rect.top)
        f: int = x // self._square_size
        r: int = y // self._square_size
        return self._get_piece_idx(r, f)