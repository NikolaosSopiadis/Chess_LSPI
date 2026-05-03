from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import pygame as pg
import pygame_gui as pgg

from gui.view.main_window import MainWindow
from chess_core.piece import Piece as p     
from chess_core.board import Board
from chess_core.move import (
    Move,
    F_CASTLE,
    PROMO_NONE,
    PROMO_QUEEN,
    PROMO_ROOK,
    PROMO_BISHOP,
    PROMO_KNIGHT,
)

from chess_rl.agents.base import Agent
from chess_rl.agents.registry import HUMAN, player_options, make_player

from dataclasses import dataclass

# type-only, no runtime imports
# if TYPE_CHECKING:

@dataclass(frozen=True, slots=True)
class MoveRecord:
    move: Move
    mover_white: bool
    notation: str
    
class Controller:
    
    STOPPED: int = 0
    RUNNING: int = 1
    
    def __init__(self, ranks: int = 8, files: int = 8) -> None:
        self._state: int = self.RUNNING
        self._ranks: int = ranks
        self._files: int = files
        
        self._view: MainWindow = MainWindow(self, 800*2, 600*2, "Chess")
        self._model: Board     = Board(ranks, files)
        
        self._white_player_id: str = HUMAN
        self._black_player_id: str = HUMAN
        self._white_agent: Agent | None = None
        self._black_agent: Agent | None = None

        self._agent_move_delay_ms: int = 250
        self._last_agent_move_ms: int = 0
        
        self._grid_size = self._ranks * self._files

        self._move_history: list[MoveRecord] = []
        self._last_move: Move | None = None
        
        self._sync_sidebar()

    def get_state(self) -> int:
        return self._state
    
    def update_state(self, state: int) -> None:
        self._state = state
        
    def get_ranks(self) -> int:
        return self._ranks
    
    def get_files(self) -> int:
        return self._files
    
    # TODO: Remove this and interact directly with controller
    def get_view(self) -> MainWindow:
        return self._view
    
    def get_pieces_on_board(self) -> Sequence[int]:
        return self._model.get_board()
    
    def get_piece_sprite(self, piece:int) -> str | None:
        color: str = "w" if p.is_white(piece) else "b"
        folder_path: str = "assets/pieces/"

        match p.piece_type(piece):
            case p.NONE:
                return None
            case p.PAWN:
                piece_type: str = "pawn-" + color + ".svg"
            case p.KNIGHT:
                piece_type: str = "knight-" + color + ".svg"
            case p.BISHOP:
                piece_type: str = "bishop-" + color + ".svg"
            case p.ROOK:
                piece_type: str = "rook-" + color + ".svg"
            case p.QUEEN:
                piece_type: str = "queen-" + color + ".svg"
            case p.KING:
                piece_type: str = "king-" + color + ".svg"
            case _:
                raise FileNotFoundError(f"Could find corresponding asset for piece: {piece}")
        return folder_path + piece_type

    def handle_event(self, event: pg.event.Event) -> None:
        
        self._view.manager_process_events(event)
        
        match event.type:
            case pg.QUIT:
                self._state = self.STOPPED
            
            case pg.KEYDOWN:
                
                match event.key:
                    case pg.K_q:
                        self._state = self.STOPPED
            
            case pg.MOUSEMOTION:
                self._view.on_mouse_move(event.pos)
                
            case pg.MOUSEBUTTONDOWN:
                self._view.on_mouse_down(event.pos)
                
            case pg.MOUSEBUTTONUP:
                self._view.on_mouse_up(event.pos)
         
            case pgg.UI_BUTTON_PRESSED:
                sb = self._view._sidebar

                if event.ui_element == sb._new_game_button:
                    self.new_game(
                        white_player_id=sb.get_white_player_id(),
                        black_player_id=sb.get_black_player_id(),
                    )

                elif event.ui_element == sb._load_fen_button:
                    self.new_game(
                        sb.get_fen_text(),
                        white_player_id=sb.get_white_player_id(),
                        black_player_id=sb.get_black_player_id(),
                    )

            case pgg.UI_TEXT_ENTRY_FINISHED:
                sb = self._view._sidebar
                if event.ui_element == sb._fen_entry:
                    self.new_game(
                        sb.get_fen_text(),
                        white_player_id=sb.get_white_player_id(),
                        black_player_id=sb.get_black_player_id(),
                    )

            case pg.VIDEORESIZE:
                x, y = event.size
                self._view.on_resize(x,y)
                    
    # For the agent: never call move_piece; always choose a full Move.
    def move_piece(self, source: int, destination: int) -> bool:
        """Attempt to move a piece from source square to destination square

        Args:
            source (int): source square index
            destination (int): destination square index

        Returns:
            bool: True if success, False if illegal move
        """
        cands = self.get_moves_to(source, destination, legal=True)
        if not cands:
            return False
        if len(cands) != 1:
            # promotion ambiguity etc. — GUI should call make_move() with chosen Move
            return False
        return self.make_move(cands[0])
    
    # TODO: SOS Cache legal moves since they are asked multiple times per second (each frame a square is highlighted)
    # def get_legal_moves(self, square: int) -> list[int]:
        # file, rank = self._model.idx_to_f_r(square)
        # return self._model.get_legal_moves(file, rank)
        # return self._model.get_legal_moves(square)
    
    def get_moves(self, src: int, legal: bool = True) -> list[Move]:
        if legal:
            return self._model.get_legal_moves(src)
        else:
            return self._model.get_pseudolegal_moves(src)

    def get_move_dests(self, src: int, legal: bool = True) -> list[int]:
        return [m.dst_square for m in self.get_moves(src, legal=legal)]
    
    def is_white_to_move(self) -> bool:
        return self._model.get_is_white_to_move()

    def is_friendly_piece(self, piece: int) -> bool:
        return piece != p.NONE and (p.is_white(piece) == self.is_white_to_move())

    def get_moves_to(self, src: int, dst: int, *, legal: bool = True) -> list[Move]:
        return [m for m in self.get_moves(src, legal=legal) if m.dst_square == dst]

    def make_move(self, move: Move) -> bool:
        mover_white = self._model.get_is_white_to_move()
        notation = self._format_move_notation(move)

        ok = self._model.make_move(move)
        if ok:
            self._last_move = move
            self._move_history.append(
                MoveRecord(
                    move=move,
                    mover_white=mover_white,
                    notation=notation,
                )
            )
            self._sync_sidebar()

        return ok

    def get_game_end_state(self) -> tuple[bool, str]:
        return self._model.game_end_state()

    def _castling_string(self) -> str:
        r = self._model._castling_rights  # bitmask
        s = ""
        if r & self._model.WHITE_CASTLE_KINGSIDE:  s += "K"
        if r & self._model.WHITE_CASTLE_QUEENSIDE: s += "Q"
        if r & self._model.BLACK_CASTLE_KINGSIDE:  s += "k"
        if r & self._model.BLACK_CASTLE_QUEENSIDE: s += "q"
        return s if s else "-"

    def get_debug_text(self) -> str:
        b = self._model
        turn = "White" if b.get_is_white_to_move() else "Black"

        ep = "-"
        if b._en_passant_target is not None:
            ep = b.idx_to_algebraic(b._en_passant_target)

        rep = b._rep_counts.get(b._zkey, 0)

        return (
            f"FEN:\n{b.to_fen()}\n"
            f"\nWhite player: {self._white_player_id}\n"
            f"Black player: {self._black_player_id}\n"
            f"\nTurn: {turn}\n"
            f"Castling: {self._castling_string()}  (mask={b._castling_rights:04b})\n"
            f"EP: {ep}\n"
            f"Halfmove: {b._halfmove_clock}\n"
            f"Zobrist: {b._zkey:#018x}\n"
            f"Repetition count (current): {rep}\n"
        )

    def _sync_sidebar(self, *, override_status: str | None = None) -> None:
        done, reason = self._model.game_end_state()
        status = override_status if override_status is not None else ("game over: " + reason if done else "playing")
        self._view._sidebar.set_status(status)
        self._view._sidebar.set_history(self.get_move_history_text())
        self._view._sidebar.set_debug(self.get_debug_text())

    def new_game(
        self,
        fen: str | None = None,
        *,
        white_player_id: str | None = None,
        black_player_id: str | None = None,
    ) -> bool:
        if white_player_id is not None or black_player_id is not None:
            ok = self.set_players(
                white_player_id or self._white_player_id,
                black_player_id or self._black_player_id,
            )
            if not ok:
                return False

        try:
            if fen and fen.strip():
                self._model.init_board(fen.strip())
            else:
                self._model.init_board()
        except ValueError as e:
            self._sync_sidebar(override_status=f"bad FEN: {e}")
            return False

        self._move_history.clear()
        self._last_move = None

        self._last_agent_move_ms = pg.time.get_ticks()

        self._view._board.on_position_changed()
        self._sync_sidebar()
        return True

    def get_player_options(self) -> list[str]:
        return player_options()

    def set_players(self, white_player_id: str, black_player_id: str) -> bool:
        try:
            white_agent = make_player(white_player_id)
            black_agent = make_player(black_player_id)
        except Exception as e:
            self._sync_sidebar(override_status=f"player error: {e}")
            return False

        self._white_player_id = white_player_id
        self._black_player_id = black_player_id
        self._white_agent = white_agent
        self._black_agent = black_agent
        return True

    def is_human_turn(self) -> bool:
        return self._current_agent() is None

    def _current_agent(self) -> Agent | None:
        return self._white_agent if self._model.get_is_white_to_move() else self._black_agent

    def update(self) -> None:
        """
        Called once per frame. Lets agent players make a move when it is their turn.
        """
        done, _reason = self._model.game_end_state()
        if done:
            return

        agent = self._current_agent()
        if agent is None:
            return

        now = pg.time.get_ticks()
        if now - self._last_agent_move_ms < self._agent_move_delay_ms:
            return

        self._last_agent_move_ms = now
        self._make_agent_move(agent)

    def _make_agent_move(self, agent: Agent) -> None:
        try:
            move = agent.pick_move(self._model)
        except Exception as e:
            self._sync_sidebar(override_status=f"agent error: {agent.info.name}: {e}")
            return

        ok = self.make_move(move)
        if not ok:
            self._sync_sidebar(override_status=f"agent illegal move: {agent.info.name}")
            return

        self._view._board.on_move_committed()

    def get_last_move(self) -> Move | None:
        return self._last_move

    def _format_move_notation(self, move: Move) -> str:
        # Simple coordinate notation, not SAN.
        # Examples: e2e4, e7e8=Q, O-O, O-O-O
        if move.flags & F_CASTLE:
            return "O-O" if (move.dst_square & 7) > (move.src_square & 7) else "O-O-O"

        s = self._model.idx_to_algebraic(move.src_square) + self._model.idx_to_algebraic(move.dst_square)

        if move.promotion != PROMO_NONE:
            promo = {
                PROMO_QUEEN: "Q",
                PROMO_ROOK: "R",
                PROMO_BISHOP: "B",
                PROMO_KNIGHT: "N",
            }.get(move.promotion, "?")
            s += f"={promo}"

        return s

    def get_move_history_text(self) -> str:
        if not self._move_history:
            return "(no moves yet)"

        lines: list[str] = []

        # Group as:
        # 1. e2e4 e7e5
        # 2. g1f3 ...
        for i in range(0, len(self._move_history), 2):
            move_no = i // 2 + 1
            white_move = self._move_history[i].notation
            black_move = self._move_history[i + 1].notation if i + 1 < len(self._move_history) else ""
            lines.append(f"{move_no}. {white_move} {black_move}".rstrip())

        return "\n".join(lines)