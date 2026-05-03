from __future__ import annotations
from typing import TYPE_CHECKING
import pygame as pg
import pygame_gui as pgg
import html

if TYPE_CHECKING:
    from gui.controller.controller import Controller

DropdownOption = str | tuple[str, str]

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

        self._wrap_chars = max(12, min(42, (width - 2 * self.PAD) // 9))

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

        self._white_player_label = pgg.elements.UILabel(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 24),
            text="White",
            manager=manager,
            container=self._sidebar,
        )
        y += 24

        opts: list[DropdownOption] = []
        opts.extend(self._ctrl.get_player_options())

        self._white_player_dropdown = pgg.elements.UIDropDownMenu(
            options_list=opts,
            starting_option=opts[0],
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 34),
            manager=manager,
            container=self._sidebar,
        )
        y += 34 + self.GAP

        self._black_player_label = pgg.elements.UILabel(
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 24),
            text="Black",
            manager=manager,
            container=self._sidebar,
        )
        y += 24

        self._black_player_dropdown = pgg.elements.UIDropDownMenu(
            options_list=opts,
            starting_option=opts[0],
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), 34),
            manager=manager,
            container=self._sidebar,
        )
        y += 34 + self.GAP

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
            html_text=_to_html_lines("Debug:\n(press New Game or Load FEN)", self._wrap_chars),
            relative_rect=pg.Rect(self.PAD, y, max(1, width - 2 * self.PAD), debug_h),
            manager=manager,
            container=self._sidebar,
        )

    # --- API used by Controller ---
    def set_status(self, text: str) -> None:
        self._status.set_text(f"Status: {text}")

    def set_debug(self, text: str) -> None:
        self._debug_box.set_text(_to_html_lines(text, self._wrap_chars))

    def get_fen_text(self) -> str:
        return self._fen_entry.get_text()

    # --- Layout ---
    def resize(self, x0: int, y0: int, new_width: int, new_height: int) -> None:
        self._x0 = x0
        self._y0 = y0
        self._width = new_width
        self._height = new_height

        self._wrap_chars = max(12, min(42, (new_width - 2 * self.PAD) // 9))

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

        self._white_player_label.set_relative_position((self.PAD, y))
        self._white_player_label.set_dimensions((w, 24))
        y += 24

        self._white_player_dropdown.set_relative_position((self.PAD, y))
        self._white_player_dropdown.set_dimensions((w, 34))
        y += 34 + self.GAP

        self._black_player_label.set_relative_position((self.PAD, y))
        self._black_player_label.set_dimensions((w, 24))
        y += 24

        self._black_player_dropdown.set_relative_position((self.PAD, y))
        self._black_player_dropdown.set_dimensions((w, 34))
        y += 34 + self.GAP

        self._fen_entry.set_relative_position((self.PAD, y))
        self._fen_entry.set_dimensions((w, 34))
        y += 34 + self.GAP

        self._load_fen_button.set_relative_position((self.PAD, y))
        self._load_fen_button.set_dimensions((w, 36))
        y += 36 + self.GAP

        debug_h = max(80, new_height - y - self.PAD)
        self._debug_box.set_relative_position((self.PAD, y))
        self._debug_box.set_dimensions((w, debug_h))

    def _dropdown_value(self, dropdown) -> str:
        value = dropdown.selected_option
        if isinstance(value, tuple):
            return str(value[0])
        return str(value)

    def get_white_player_id(self) -> str:
        return self._dropdown_value(self._white_player_dropdown)

    def get_black_player_id(self) -> str:
        return self._dropdown_value(self._black_player_dropdown)


def _wrap_debug_line(line: str, width: int) -> list[str]:
    if len(line) <= width:
        return [line]

    out: list[str] = []
    s = line

    while len(s) > width:
        cut_space = s.rfind(" ", 0, width + 1)
        cut_slash = s.rfind("/", 0, width + 1)
        cut = max(cut_space, cut_slash)

        if cut < max(4, width // 3):
            cut = width
            out.append(s[:cut])
            s = s[cut:]
        else:
            if s[cut] == "/":
                out.append(s[:cut + 1])
                s = s[cut + 1:]
            else:
                out.append(s[:cut])
                s = s[cut:].lstrip()

    if s:
        out.append(s)

    return out


def _to_html_lines(text: str, wrap_chars: int = 32) -> str:
    lines: list[str] = []

    for line in text.splitlines():
        if line == "":
            lines.append("")
            continue

        for wrapped in _wrap_debug_line(line, wrap_chars):
            lines.append(html.escape(wrapped))

    return "<br>".join(lines)