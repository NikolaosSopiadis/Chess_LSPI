from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg

if TYPE_CHECKING:
    from gui.controller.controller import Controller


def _to_html_lines(text: str) -> str:
    # pygame_gui UITextBox expects "html-like" text; <br> works for newlines.
    return "<br>".join(text.splitlines())


class Sidebar:
    PAD = 16
    GAP = 10

    def __init__(self, x0: int, y0: int, width: int, height: int,
                 manager: pgg.UIManager, controller: Controller) -> None:
        self._x0 = x0
        self._y0 = y0
        self._width = width
        self._height = height
        self._ctrl = controller

        self._sidebar_rect = pg.Rect(x0, y0, width, height)
        self._sidebar = pgg.elements.UIPanel(
            relative_rect=self._sidebar_rect,
            starting_height=y0,
            manager=manager
        )

        # Layout (relative to panel)
        y = 10

        self._title = pgg.elements.UILabel(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 28),
            text="Game Controls",
            manager=manager,
            container=self._sidebar,
        )
        y += 28 + self.GAP

        self._status = pgg.elements.UILabel(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 24),
            text="Status: playing",
            manager=manager,
            container=self._sidebar,
        )
        y += 24 + self.GAP

        self._new_game_button = pgg.elements.UIButton(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 40),
            text="New Game",
            manager=manager,
            container=self._sidebar,
        )
        y += 40 + self.GAP

        self._fen_entry = pgg.elements.UITextEntryLine(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 34),
            manager=manager,
            container=self._sidebar,
        )
        self._fen_entry.set_text_length_limit(200)
        self._fen_entry.set_text("")  # paste FEN here
        y += 34 + self.GAP
        
        self._load_fen_button = pgg.elements.UIButton(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 36),
            text="Load FEN",
            manager=manager,
            container=self._sidebar,
        )
        y += 36 + self.GAP

        # Debug panel fills remaining space
        debug_h = max(80, height - y - self.PAD)
        self._debug_box = pgg.elements.UITextBox(
            html_text=_to_html_lines("Debug:\n(press New Game or Load FEN)"),
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), debug_h),
            manager=manager,
            container=self._sidebar,
        )

    # --- API used by Controller ---
    def set_status(self, text: str) -> None:
        self._status.set_text(f"Status: {text}")

    def set_debug(self, text: str) -> None:
        self._debug_box.set_text(_to_html_lines(text))

    def get_fen_text(self) -> str:
        return self._fen_entry.get_text()

    # --- Layout ---
    def resize(self, x0: int, y0: int, new_width: int, new_height: int) -> None:
        self._x0 = x0
        self._y0 = y0
        self._width = new_width
        self._height = new_height

        self._sidebar.set_relative_position((x0, y0))
        self._sidebar.set_dimensions((new_width, new_height))

        y = 10
        w = max(1, new_width - 2 * self.PAD)

        self._title.set_relative_position((self.PAD, y))
        self._title.set_dimensions((w, 28))
        y += 28 + self.GAP

        self._status.set_relative_position((self.PAD, y))
        self._status.set_dimensions((w, 24))
        y += 24 + self.GAP

        self._new_game_button.set_relative_position((self.PAD, y))
        self._new_game_button.set_dimensions((w, 40))
        y += 40 + self.GAP

        self._fen_entry.set_relative_position((self.PAD, y))
        self._fen_entry.set_dimensions((w, 34))
        y += 34 + self.GAP

        self._load_fen_button.set_relative_position((self.PAD, y))
        self._load_fen_button.set_dimensions((w, 36))
        y += 36 + self.GAP

        debug_h = max(80, new_height - y - self.PAD)
        self._debug_box.set_relative_position((self.PAD, y))
        self._debug_box.set_dimensions((w, debug_h))
