from __future__ import annotations
from typing import TYPE_CHECKING
from enum import Enum
import pygame as pg
import pygame_gui as pgg
import numpy as np
import numpy.typing as npt

from chess_core.piece import Piece as p

# type-only, no runtime imports (to avoid circular dependency)
if TYPE_CHECKING:
    from gui.controller.controller import Controller  

class UIState(Enum):
    IDLE     = 0
    SELECTED = 1
    DRAGGING = 2
    
class BoardWindow:
    def __init__(self, controller: Controller, x0: int, y0: int,
                 width: int, height: int, manager) -> None:

        self._ctrl:       Controller = controller
        self._manager: pgg.UIManager = manager

        # Cache sprites to increase fps
        self._sprite_cache: dict[tuple[int, int], pg.Surface] = {} # (square_size, piece_type) -> piece_sprite
        self._max_sprite_sizes_cached:   int = 3
        self._cached_sizes_in_use: list[int] = [] # LRU of sizes
        
        self._width:          int = width
        self._height:         int = height
        self._x0:             int = x0
        self._y0:             int = y0
        self._board_flipped: bool = False
        
        # self.flip_board()
        
        self._ranks:       int = self._ctrl.get_ranks()
        self._files:       int = self._ctrl.get_files()
        self._square_size: int = self._height // max(self._files, self._ranks)
        
        self._ui_state: UIState               = UIState.IDLE
        self._selected_idx: int               = -1       # board index, -1 if none
        self._hover_idx: int                  = -1       # board index under cursor, -1 if none
        self._drag_origin_idx: int            = -1       # where drag started
        self._legal_dests: set[int]           = set()    # cached destinations for selected piece
        self._mouse_down_pos: tuple[int, int] = (-1, -1) # (-1, -1) if none else (file, rank)

        self._clear_selection_on_mouse_up: bool = False
        
        # Try to decrease resource utilization by not redrawing when not necessary
        self._needs_redraw:  bool = True         
        self._board_dirty:   bool = True  # squares/flip/size changed
        self._pieces_dirty:  bool = True  # pieces changed (on move)
        self._overlay_dirty: bool = True  # hover/selection changed

        self._hovers_over:       tuple[int, int] = (-1,-1) # (-1, -1) if none else (file, rank)
        self._selected:          tuple[int, int] = (-1,-1) # (-1, -1) if none else (file, rank)
        self._picked_up_piece:               int = -1 # -1 if none else idx of picked up piece
        self._mouse_clicked:                bool = False
        self._mouse_clicked_pos: tuple[int, int] = (-1, -1) # (-1, -1) if none else (file, rank)
        self._mouse_pos:     tuple[float, float] = (-1, -1) # (-1, -1) if none else (x, y)
        
        self._board_rect: pg.Rect = pg.Rect(x0, y0, width, height)
        
        self._board_surface:    pg.Surface = pg.Surface((width, height)).convert()
        self._pieces_surface:   pg.Surface = pg.Surface((width, height), pg.SRCALPHA)
        self._overlay_surface:  pg.Surface = pg.Surface((width, height), pg.SRCALPHA)
        self._composed_surface: pg.Surface = pg.Surface((width, height)).convert()

        self._board: pgg.elements.UIImage = pgg.elements.UIImage(
            relative_rect = self._board_rect,
            image_surface = self._board_surface,
            manager = manager,
        )
        
    def _get_sprite(self, piece: int) -> pg.Surface | None:

        # Empty space has no sprite
        if piece == p.NONE:
            return None
        
        # Try cache
        key:      tuple[int, int] = (self._square_size, piece)
        cached_sprite: pg.Surface | None = self._sprite_cache.get(key)
        if cached_sprite != None:
            return cached_sprite
        
        # Cache miss
        path: str | None = self._ctrl.get_piece_sprite(piece)

        # None when empty, will likely never enter since we check for empty at the start
        if path == None:
            return None

        sprite: pg.Surface = pg.image.load_sized_svg(path, (self._square_size, self._square_size)).convert_alpha() 

        # Store and manage LEU of sizes
        self._sprite_cache[key] = sprite
        self._touch_size_lru(self._square_size)
        self._trim_size_variants_if_needed()
        
        return sprite
    
    def _touch_size_lru(self, size: int) -> None:
        # move 'size' to the end
        if size in self._cached_sizes_in_use:
            self._cached_sizes_in_use.remove(size)
        self._cached_sizes_in_use.append(size)

    def _trim_size_variants_if_needed(self) -> None:
        # If we exceed allowed distinct sizes, drop the oldest size entries
        while len(self._cached_sizes_in_use) > self._max_sprite_sizes_cached:
            old_size = self._cached_sizes_in_use.pop(0)
            # delete all entries of that size from the cache
            to_del = [k for k in self._sprite_cache.keys() if k[0] == old_size]
            for k in to_del:
                del self._sprite_cache[k]
    
    def draw(self) -> None:
        # Redraw only when something changes
        if not self._needs_redraw:
            return
        self._needs_redraw = False

        if self._board_dirty:
            self._rebuild_chess_board_surface()
        
        if self._pieces_dirty:
            self._rebuild_pieces_surface()
        
        if self._overlay_dirty:
            self._rebuild_overlay_surface()
        
        self._composed_surface.blit(self._board_surface, (0, 0))
        self._composed_surface.blit(self._pieces_surface, (0, 0))
        self._composed_surface.blit(self._overlay_surface, (0, 0))
        
        self._board.set_image(self._composed_surface)  # update the displayed image 
        
    def _rebuild_chess_board_surface(self) -> None:
        light_color: tuple[int, int, int] = (200, 180, 160)
        dark_color:  tuple[int, int, int] = (60, 50, 40)
        # Fill board with light squares. Now I only need to draw the dark squares (half the draw calls)
        self._board_surface.fill(light_color)  

        for r in range(self._ranks):
            for f in range(self._files):
                # square:             pg.Rect = pg.Rect(f * self._square_size, r * self._square_size, self._square_size, self._square_size)
                # color: tuple[int, int, int] = light_color if (r + f) % 2 == 0 else dark_color
                # pg.draw.rect(self._board_surface, color, square)
                
                # Skip light squares
                if (r + f) % 2 == 0:
                    continue
                
                square: pg.Rect = pg.Rect(f * self._square_size, r * self._square_size, self._square_size, self._square_size)
                pg.draw.rect(self._board_surface, dark_color, square)
        self._board_dirty = False
                    
    def _rebuild_pieces_surface(self) -> None:
        # Clear layer
        self._pieces_surface.fill((0,0,0,0))
        pieces: npt.NDArray[np.uint8] = self._ctrl.get_pieces_on_board()
        
        debug_text: bool = True
        if debug_text:
            font: pg.font.Font = pg.font.Font(None, 20)

        for r in range(self._ranks):
            for f in range(self._files):
                square: pg.Rect = pg.Rect(f * self._square_size, r * self._square_size, self._square_size, self._square_size)
                piece_idx: int = self._get_idx(f, r)

                if debug_text:
                    msg: str = f"idx = {piece_idx}\nx,y = ({f},{r})\n\npiece = {pieces[piece_idx]}"
                    text:  pg.Surface = font.render(msg, True, (255, 255, 255))
                    self._pieces_surface.blit(text, square)

                # don't draw picked up pieces, as they are drawn as an overlay
                if piece_idx == self._picked_up_piece:
                    continue

                self._draw_piece(pieces[piece_idx], square)
        self._pieces_dirty = False
    
    def _rebuild_overlay_surface(self) -> None:
        # Clear
        self._overlay_surface.fill((0, 0, 0, 0))

        self._draw_hover()
        self._draw_selected()
        self._draw_picked_up_piece()
        
        self._overlay_dirty = False
        
                    
    def flip_board(self) -> None:
        self._board_flipped = not self._board_flipped
        self._board_dirty   = True
        self._pieces_dirty  = True
        self._needs_redraw  = True
        
    def _get_idx(self, file: int, rank: int) -> int:
        if self._board_flipped:
            piece_idx: int = self._files - file - 1 + (rank * self._files)
        else:
            piece_idx: int = file + ((self._ranks - rank - 1) * self._files)
        
        return piece_idx
    
    def _draw_piece(self, piece: int, square: pg.Rect) -> None:
        sprite: pg.Surface | None = self._get_sprite(piece)

        # No piece to draw
        if sprite == None:
            return
        
        self._pieces_surface.blit(sprite, square)
        
    def set_mouse_pos(self, pos: tuple[float, float]) -> None:
        self._mouse_pos = pos

    def check_hover(self) -> None:
        # if mouse is outside of board, then no piece is selected
        if not self._board_rect.collidepoint(self._mouse_pos):
            if self._hovers_over != (-1, -1):
                self._hovers_over = (-1, -1)
                self._needs_redraw  = True
                self._overlay_dirty = True
            return 
        
        # This updates the moving piece sprite only when entering a different square
        # thus making the motion choppy
        
        # new_hover: tuple[int, int] = self._file_rank_from_mouse_pos(self._mouse_pos)
        # if new_hover != self._hovers_over:
        #     self._hovers_over   = new_hover
        #     self._needs_redraw  = True
        #     self._overlay_dirty = True

        new_hover: tuple[int, int] = self._file_rank_from_mouse_pos(self._mouse_pos)
        self._hovers_over   = new_hover
        self._needs_redraw  = True
        self._overlay_dirty = True
        
    def _get_idx_from_mouse_pos(self, mouse_pos: tuple[float, float]) -> int:
        f, r = self._file_rank_from_mouse_pos(mouse_pos) 
        if f < 0 or r < 0 or f >= self._files or r >= self._ranks:
            return -1
        return self._get_idx(f, r)
    
    def _file_rank_from_mouse_pos(self, mouse_pos: tuple[float, float]) -> tuple[int, int]:
        x: int = int(mouse_pos[0] - self._board_rect.left)
        y: int = int(mouse_pos[1] - self._board_rect.top)

        f: int = x // self._square_size
        r: int = y // self._square_size
        return f,r
    
    def _draw_hover(self) -> None:
        if self._hover_idx == -1:
            return
        f, r = self._idx_to_f_r(self._hover_idx)
        hover_color: tuple[int, int, int, int] = (200, 200, 0, 128)
        self._draw_overlay_square(f, r, hover_color)
    
    # def _draw_hover(self) -> None:
    #     f, r = self._hovers_over
    #     if f == -1 or r == -1:
    #         return
    #     hover_color: tuple[int, int, int, int] = (200, 200, 0, 128)
    #     self._draw_overlay_square(f, r, hover_color)
        
    def select_square(self) -> None:
        # Ignore clicks outside the board
        f, r = self._hovers_over
        if f == -1 or r == -1:
            self._selected = (-1, -1)
            return
        
        # # If mouse_up != mouse_down, ignore
        # f_clicked, r_clicked = self._mouse_clicked_pos
        # if f != f_clicked or r != r_clicked:
        #     self._selected = (-1, -1)
        #     return

        # If clicked on active square, remove selection
        f_s, r_s = self._selected
        if f == f_s and r == r_s:
            self._selected      = (-1, -1)
            self._needs_redraw  = True
            self._overlay_dirty = True
            return

        # If a square is selected, attempt to move the piece
        if f_s != -1 and f_s != -1:
            src = self._get_idx(f_s, r_s)
            dst = self._get_idx(f, r)
            self._ctrl.move_piece(src, dst)
            self._selected      = (-1, -1)
            self._needs_redraw  = True
            self._overlay_dirty = True
            self._pieces_dirty  = True
            return

        self._selected      = (f, r)
        self._needs_redraw  = True
        self._overlay_dirty = True

    def _draw_overlay_square(self, file: int, rank: int, color: tuple[int, int, int, int]) -> None:
        square: pg.Rect = pg.Rect(file * self._square_size, rank * self._square_size, self._square_size, self._square_size)
        pg.draw.rect(self._overlay_surface, color, square)

    def _draw_selected(self) -> None:
        f, r = self._selected
        if f == -1 or r == -1:
            return

        # Highlight the selected square
        selected_color: tuple[int, int, int, int] = (200, 100, 0, 128)
        self._draw_overlay_square(f, r, selected_color)
        
        # Draw legal moves
        legal_move_color: tuple[int, int, int, int] = (50, 255, 50, 128)
        square_idx: int = self._get_idx(f, r)
        legal_moves = self._ctrl.get_move_dests(square_idx)
        for lm in legal_moves:
            lm_f, lm_r = self._idx_to_f_r(lm)
            self._draw_overlay_square(lm_f, lm_r, legal_move_color)
            
    # TODO: Super important! f, r in board_window refer to the x,y relataive cordinates and not the file, rank in board
    # I need to fix it so view uses the absolute f,r from board       
    # Or at leaste re-write them in x,y to avoid confusion
    def _idx_to_f_r(self, idx: int) -> tuple[int, int]:
        r: int = idx // self._files
        f: int = idx % self._files
        
        if self._board_flipped:
            f = self._files - f -1
        else:
            r = self._ranks - r - 1
            
        return f, r
        
    def _pick_piece_up(self) -> None:
        f, r = self._hovers_over
        if f == -1 or r == -1:
            self._picked_up_piece = -1
            return
            
        self._picked_up_piece   = self._get_idx(f, r)
        self._mouse_clicked_pos = self._hovers_over
        
    def _draw_picked_up_piece(self) -> None:
        if self._mouse_clicked == False or self._picked_up_piece == -1:
            return
        
        x, y = self._mouse_pos
        square: pg.Rect = pg.Rect(x - self._square_size/2, y - self._square_size/2, self._square_size, self._square_size)
        piece:      int = self._ctrl.get_pieces_on_board()[self._picked_up_piece]

        sprite: pg.Surface | None = self._get_sprite(piece)
        # No piece to draw
        if sprite == None:
            return
        self._overlay_surface.blit(sprite, square)
        
        
    def set_mouse_clicked(self, clicked: bool) -> None:
        self._mouse_clicked = clicked
        
        if clicked:
            self._pick_piece_up()
        else:
            self._place_piece_down()

        self._needs_redraw  = True
        self._pieces_dirty  = True
        self._overlay_dirty = True
            
    def _place_piece_down(self) -> None:
        self._picked_up_piece = -1
        f_src, r_src     = self._mouse_clicked_pos
        f_dst, r_dst     = self._hovers_over
        source:      int = self._get_idx(f_src, r_src)
        destination: int = self._get_idx(f_dst, r_dst)
        self._ctrl.move_piece(source, destination)

    def resize(self, new_width: int, new_height: int) -> None:
        self._width  = new_width
        self._height = new_height
        # new_sq: int  = new_height // max(self._files, self._ranks)
        new_sq = max(1, self._height // max(1, max(self._files, self._ranks)))

        self._square_size = new_sq
        # mark layers dirty; sprites will be (re)cached lazily
        self._board_dirty   = True
        self._pieces_dirty  = True
        self._overlay_dirty = True
        self._needs_redraw  = True
    
        self._board_rect:          pg.Rect = pg.Rect(self._x0, self._y0, new_width, new_height)
        self._board_surface:    pg.Surface = pg.Surface((new_width, new_height)).convert()
        self._pieces_surface:   pg.Surface = pg.Surface((new_width, new_height), pg.SRCALPHA)
        self._overlay_surface:  pg.Surface = pg.Surface((new_width, new_height), pg.SRCALPHA)
        self._composed_surface: pg.Surface = pg.Surface((new_width, new_height)).convert()
        self._board.set_dimensions((new_width, new_height))

    def on_mouse_down(self, mouse_pos: tuple[float, float]) -> None:
        self._mouse_clicked   = True
        self._mouse_pos = mouse_pos
        idx: int              = self._get_idx_from_mouse_pos(mouse_pos)
        self._mouse_down_pos  = self._file_rank_from_mouse_pos(mouse_pos)
        self._drag_origin_idx = idx
        self._picked_up_piece = self._drag_origin_idx

        if self._selected[0] != -1 and self._selected[1] != -1 and not self._is_friendly(idx):
            # If a piece is selected, make move
            selected_idx: int = self._get_idx(self._selected[0], self._selected[1])
            self._ctrl.move_piece(selected_idx, idx)
            self._selected = -1, -1

        elif idx != -1 and self._is_friendly(idx):
            # If the clicked piece is friendly, select it
            selected_idx: int = self._get_idx(self._selected[0], self._selected[1])
            if selected_idx != idx:
                self._selected   = self._idx_to_f_r(idx)
                self._legal_dests = set(self._ctrl.get_move_dests(idx))
                self._clear_selection_on_mouse_up = False
            else:
                # If clicked on an already selected piece, remove the selection on mouse up
                self._clear_selection_on_mouse_up = True
        else:
            # clear selection
            self._legal_dests.clear()
            self._selected = -1, -1
        
        self._pieces_dirty  = True
        self._overlay_dirty = True
        self._needs_redraw  = True
            
    def on_mouse_move(self, mouse_pos: tuple[float, float]) -> None:
        self._mouse_pos = mouse_pos
        new_hover: int  = self._get_idx_from_mouse_pos(mouse_pos)

        if new_hover != self._hover_idx:
            self._hover_idx     = new_hover
            self._overlay_dirty = True
            self._needs_redraw  = True

        self._overlay_dirty   = True
        self._needs_redraw    = True

    def on_mouse_up(self, mouse_pos: tuple[float, float]) -> None:
        self._mouse_clicked = False
        dst: int = self._get_idx_from_mouse_pos(mouse_pos)

        if dst == -1:
            return
        
        if self._ctrl.get_pieces_on_board()[self._picked_up_piece] == p.NONE:
            return

        if self._mouse_down_pos == self._file_rank_from_mouse_pos(mouse_pos):
            if self._clear_selection_on_mouse_up:
                self._selected = -1, -1
        else:
            self._ctrl.move_piece(self._picked_up_piece, dst)
            self._selected = -1, -1

        self._legal_dests.clear()
        self._picked_up_piece = -1
        self._pieces_dirty    = True
        self._overlay_dirty   = True
        self._needs_redraw    = True
                            
    def _is_friendly(self, idx: int) -> bool:
        if idx < 0 or idx >= self._ranks * self._files: 
            return False
        piece = int(self._ctrl.get_pieces_on_board()[idx])
        if piece == p.NONE: 
            return False
        return self._ctrl.is_friendly_piece(piece)