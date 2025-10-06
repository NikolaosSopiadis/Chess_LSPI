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
        
        
        self._hovers_over:       tuple[int, int] = (-1,-1) # (-1, -1) if none else (file, rank)
        self._selected:          tuple[int, int] = (-1,-1) # (-1, -1) if none else (file, rank)
        self._picked_up_piece:               int = -1 # -1 if none else idx of picked up piece
        self._mouse_clicked:                bool = False
        self._mouse_clicked_pos: tuple[int, int] = (-1, -1) # (-1, -1) if none else (file, rank)
        self._mouse_pos:     tuple[float, float] = (-1, -1) # (-1, -1) if none else (x, y)
        
        self._board_rect: pg.Rect = pg.Rect(x0, y0, width, height)
        
        self._board_surface:   pg.Surface = pg.Surface((width, height))
        
        self._overlay_surface: pg.Surface = pg.Surface((width, height), pg.SRCALPHA)

        self._board: pgg.elements.UIImage = pgg.elements.UIImage(
            relative_rect = self._board_rect,
            image_surface = self._board_surface,
            manager = manager,
        )
    
    def draw(self) -> None:
        # clear previous overlay
        self._overlay_surface.fill((0, 0, 0, 0))
        
        self._draw_chess_board()
        self._draw_picked_up_piece()
        self._draw_hover()
        self._draw_selected()
        self._board.set_image(self._board_surface)  # update the displayed image 
        
    def _draw_chess_board(self) -> None:
        light_color: tuple[int, int, int] = (200, 180, 160)
        dark_color:  tuple[int, int, int] = (60, 50, 40)
        
        pieces: npt.NDArray[np.uint8] = self._ctrl.get_pieces_on_board()

        for r in range(self._ranks):
            for f in range(self._files):
                square:             pg.Rect = pg.Rect(f * self._square_size, r * self._square_size, self._square_size, self._square_size)
                color: tuple[int, int, int] = light_color if (r + f) % 2 == 0 else dark_color
                pg.draw.rect(self._board_surface, color, square)
                
                piece_idx: int = self._get_idx(f, r)

                font: pg.font.Font = pg.font.Font(None, 40)
                # x,y not drawn correctly for flipped board
                msg:           str = f"idx = {piece_idx}\nx,y = ({f},{r})\n\npiece = {pieces[piece_idx]}"
                text:   pg.Surface = font.render(msg, True, (255, 255, 255))
                self._board_surface.blit(text, square)

                # don't draw picked up pieces, as they are drawn as an overlay
                if piece_idx == self._picked_up_piece:
                    continue
                self._draw_piece(pieces[piece_idx], square)
                    
    def flip_board(self) -> None:
        self._board_flipped = not self._board_flipped
        
    def _get_idx(self, file: int, rank: int) -> int:
        if self._board_flipped:
            piece_idx: int = self._files - file - 1 + (rank * self._files)
        else:
            piece_idx: int = file + ((self._ranks - rank - 1) * self._files)
        
        return piece_idx
    
    def _draw_piece(self, piece: int, square: pg.Rect) -> None:
        sprite_path: str = self._ctrl.get_piece_sprite(piece)
        
        # No piece to draw
        if sprite_path == "":
            return
        
        sprite: pg.Surface = pg.image.load_sized_svg(sprite_path, (self._square_size, self._square_size)).convert_alpha()
        self._board_surface.blit(sprite, square)
        
    def set_mouse_pos(self, pos: tuple[float, float]) -> None:
        self._mouse_pos = pos
        
    def check_hover(self) -> None:

        # if mouse is outside of board, then no piece is selected
        if not self._board_rect.collidepoint(self._mouse_pos):
            self._hovers_over = (-1, -1)
            return 
        self._hovers_over = self._file_rank_from_mouse_pos(self._mouse_pos)
        
    def _get_idx_from_mouse_pos(self, mouse_pos: tuple[float, float]) -> int:
        f, r = self._file_rank_from_mouse_pos(mouse_pos) 
        return self._get_idx(f, r)
    
    def _file_rank_from_mouse_pos(self, mouse_pos: tuple[float, float]) -> tuple[int, int]:
        x: int = int(mouse_pos[0] - self._board_rect.left)
        y: int = int(mouse_pos[1] - self._board_rect.top)

        f: int = x // self._square_size
        r: int = y // self._square_size
        return f,r
    
    def _draw_hover(self) -> None:
        f, r = self._hovers_over
        if f == -1 or r == -1:
            return
        hover_color: tuple[int, int, int, int] = (200, 200, 0, 128)
        self._draw_square(f, r, hover_color)
        
    def select_square(self) -> None:
        f, r = self._hovers_over
        if f == -1 or r == -1:
            self._selected = (-1, -1)
            return
        
        # select only if mouse_up == mouse_down
        f_clicked, r_clicked = self._mouse_clicked_pos
        if f != f_clicked or r != r_clicked:
            self._selected = (-1, -1)
            return

        # if clicked on active square, remove selection
        f_s, r_s = self._selected
        if f == f_s and r == r_s:
            self._selected = (-1, -1)
            return

        self._selected = (f, r)

    def _draw_square(self, file: int, rank: int, color: tuple[int, int, int, int]) -> None:
        square: pg.Rect = pg.Rect(file * self._square_size, rank * self._square_size, self._square_size, self._square_size)
        pg.draw.rect(self._overlay_surface, color, square)
        self._board_surface.blit(self._overlay_surface)

    def _draw_selected(self) -> None:
        f, r = self._selected
        if f == -1 or r == -1:
            return
        selected_color: tuple[int, int, int, int] = (200, 100, 0, 128)
        self._draw_square(f, r, selected_color)
        
    def _pick_piece_up(self) -> None:
        f, r = self._hovers_over
        self._picked_up_piece   = self._get_idx(f, r)
        self._mouse_clicked_pos = self._hovers_over
        print(f"hovers over: {self._hovers_over}")
        print(f"picked up piece: {self._picked_up_piece}")
        
        
    def _draw_picked_up_piece(self) -> None:
        if self._mouse_clicked == False:
            return
        
        x, y = self._mouse_pos
        square: pg.Rect = pg.Rect(x - self._square_size/2, y - self._square_size/2, self._square_size, self._square_size)
        pieces: npt.NDArray[np.uint8] = self._ctrl.get_pieces_on_board()
        self._draw_piece(pieces[self._picked_up_piece], square)
        
    def set_mouse_clicked(self, clicked: bool) -> None:
        self._mouse_clicked = clicked
        if clicked:
            self._pick_piece_up()
        else:
            self._place_piece_down()
            
    def _place_piece_down(self) -> None:
        self._picked_up_piece = -1
        f_src, r_src     = self._mouse_clicked_pos
        f_dst, r_dst     = self._hovers_over
        source:      int = self._get_idx(f_src, r_src)
        destination: int = self._get_idx(f_dst, r_dst)
        self._ctrl.move_piece(source, destination)