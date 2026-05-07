from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import random
import time
from typing import Any

from chess_core.board import Board
from chess_core.move import Move

from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.material_greedy import MaterialGreedyAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent
from chess_rl.agents.lspi_tactical import LSPITacticalAgent
from chess_rl.agents.material_minimax import MaterialMinimaxAgent

try:
    from chess_rl.agents.lspi_search import LSPISearchAgent
except Exception:
    LSPISearchAgent = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from scripts.eval_selfplay import (
    GameResult,
    make_random_opening_fen,
    play_game,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    path: str | None = None

    # Search-only options.
    search_depth: int = 2
    search_max_branch: int | None = None
    search_use_draw_safety: bool = True
    search_use_tactical_safety: bool = True


@dataclass(frozen=True)
class StartPosition:
    label: str
    fen: str
    tags: tuple[str, ...] = ()


@dataclass
class OrderedStats:
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score_sum: float = 0.0
    plies: int = 0
    reasons: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = Counter()

    @property
    def score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.score_sum / self.games

    @property
    def avg_plies(self) -> float:
        if self.games == 0:
            return 0.0
        return self.plies / self.games

@dataclass
class OpeningBehaviorStats:
    games: int = 0

    queen_moves_10: int = 0
    queen_out_10: int = 0

    castled_20: int = 0

    minor_dev_10: int = 0
    center_pawns_8: int = 0

    @property
    def avg_queen_moves_10(self) -> float:
        return self.queen_moves_10 / self.games if self.games else 0.0

    @property
    def queen_out_rate_10(self) -> float:
        return self.queen_out_10 / self.games if self.games else 0.0

    @property
    def castle_rate_20(self) -> float:
        return self.castled_20 / self.games if self.games else 0.0

    @property
    def avg_minor_dev_10(self) -> float:
        return self.minor_dev_10 / self.games if self.games else 0.0

    @property
    def avg_center_pawns_8(self) -> float:
        return self.center_pawns_8 / self.games if self.games else 0.0

def canonical_kind(kind: str) -> str:
    k = kind.lower().strip()

    if k == "random":
        return "random"

    if k in ("material", "material_greedy", "greedy_material"):
        return "material"

    if k in ("material_minimax", "minimax", "material_search"):
        return "material_search"

    if k in ("plain", "lspi_plain", "lspi_v1", "v3.0", "v4.0", "v5.0_1800", "v5.0_2000"):
        return "plain"

    if k in ("tactical", "safe", "lspi_tactical", "lspi_safe", "v3.1", "v4.1", "v5.1_1800", "v5.1_2000"):
        return "tactical"

    if k in ("search", "lspi_search", "v3.2", "v4.2", "v5.2_1800", "v5.2_2000"):
        return "search"

    raise ValueError(f"unknown model kind: {kind!r}")


def parse_bool(text: str) -> bool:
    t = text.lower().strip()

    if t in ("1", "true", "yes", "y", "on"):
        return True

    if t in ("0", "false", "no", "n", "off"):
        return False

    raise ValueError(f"invalid bool: {text!r}")


def parse_optional_int(text: str) -> int | None:
    t = text.lower().strip()

    if t in ("none", "null", "-"):
        return None

    return int(t)


def parse_opts(text: str) -> dict[str, str]:
    if not text:
        return {}

    out: dict[str, str] = {}

    for part in text.split(","):
        if not part:
            continue

        if "=" not in part:
            raise ValueError(f"bad option {part!r}; expected key=value")

        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()

    return out


def parse_model(text: str) -> ModelSpec:
    """
    Format:

      label:kind:path

    Optional fourth field:

      label:kind:path:depth=2,max_branch=8,draw=1,tactical=1

    Examples:

      v3.1:tactical:data/processed/checkpoints/foo.npz
      v3.2:search:data/processed/checkpoints/foo.npz:depth=2,max_branch=none
      v4.1:tactical:data/processed/checkpoints/bar.npz
      material:material:-
      random:random:-
    """
    parts = text.split(":", 3)

    if len(parts) < 2:
        raise ValueError(
            "model must look like label:kind:path or "
            "label:kind:path:depth=2,max_branch=8"
        )

    label = parts[0].strip()
    kind = canonical_kind(parts[1].strip())

    path: str | None = None
    opts_text = ""

    if len(parts) >= 3:
        raw_path = parts[2].strip()
        if raw_path not in ("", "-", "none", "None"):
            path = raw_path

    if len(parts) == 4:
        opts_text = parts[3].strip()

    opts = parse_opts(opts_text)

    search_depth = int(opts.get("depth", 2))
    search_max_branch = parse_optional_int(opts.get("max_branch", "none"))
    search_use_draw_safety = parse_bool(opts.get("draw", "1"))
    search_use_tactical_safety = parse_bool(opts.get("tactical", "1"))

    return ModelSpec(
        label=label,
        kind=kind,
        path=path,
        search_depth=search_depth,
        search_max_branch=search_max_branch,
        search_use_draw_safety=search_use_draw_safety,
        search_use_tactical_safety=search_use_tactical_safety,
    )


def make_agent(spec: ModelSpec, *, seed: int) -> Agent:
    if spec.kind == "random":
        return RandomAgent(seed=seed)

    if spec.kind == "material":
        return MaterialGreedyAgent(seed=seed)

    if spec.kind == "material_search":
        return MaterialMinimaxAgent(
            depth=spec.search_depth,
            seed=seed,
        )

    if spec.path is None:
        raise ValueError(f"model {spec.label!r} requires a checkpoint path")

    if not Path(spec.path).exists():
        raise FileNotFoundError(f"checkpoint not found for {spec.label}: {spec.path}")

    if spec.kind == "plain":
        return LSPIV1Agent.load(spec.path)

    if spec.kind == "tactical":
        return LSPITacticalAgent.load(spec.path)

    if spec.kind == "search":
        if LSPISearchAgent is None:
            raise RuntimeError("Could not import chess_rl.agents.lspi_search.LSPISearchAgent")

        return LSPISearchAgent.load(
            spec.path,
            depth=spec.search_depth,
            max_branch=spec.search_max_branch,
            use_draw_safety=spec.search_use_draw_safety,
            use_tactical_safety=spec.search_use_tactical_safety,
        )

    raise AssertionError(spec.kind)


def move_to_uci_like(board: Board, move: Move) -> str:
    text = board.idx_to_algebraic(move.src_square) + board.idx_to_algebraic(move.dst_square)

    if getattr(move, "promotion", 0):
        text += f"={move.promotion}"

    return text


def find_uci_move(board: Board, uci: str) -> Move:
    for move in board.get_all_legal_moves():
        if move_to_uci_like(board, move).startswith(uci):
            return move

    legal = [move_to_uci_like(board, m) for m in board.get_all_legal_moves()]
    raise ValueError(f"Could not find legal move {uci!r} in {board.to_fen()}. Legal: {legal}")


def fen_from_uci_sequence(moves: list[str]) -> str:
    board = Board()
    board.init_board(START_FEN)

    for uci in moves:
        move = find_uci_move(board, uci)
        ok = board.make_move(move)

        if not ok:
            raise ValueError(f"Move {uci!r} failed in sequence {moves}")

    return board.to_fen()


def _seq(label: str, tags: tuple[str, ...], moves: list[str]) -> StartPosition:
    return StartPosition(
        label=label,
        fen=fen_from_uci_sequence(moves),
        tags=tags,
    )


def _fen(label: str, tags: tuple[str, ...], fen: str) -> StartPosition:
    return StartPosition(
        label=label,
        fen=fen,
        tags=tags,
    )


def built_in_position_pool() -> list[StartPosition]:
    """
    Representative evaluation starts.

    These are deliberately not random. They are broad probes:
      - standard start
      - developed openings
      - open/closed/semi-closed middlegames
      - queenless positions
      - attacking positions
      - endgames, including won, drawn, mating, and imbalanced material

    Later, this can be supplemented with PGN-extracted FENs through --positions-json.
    """
    positions: list[StartPosition] = [
        _fen(
            "standard_start",
            ("start", "opening"),
            START_FEN,
        ),

        # ------------------------------------------------------------------
        # Openings / early developed positions
        # ------------------------------------------------------------------
        _seq(
            "italian_developed_uncastled",
            ("opening", "open", "developed", "uncastled"),
            [
                "e2e4", "e7e5",
                "g1f3", "b8c6",
                "f1c4", "g8f6",
                "d2d3", "f8c5",
                "c2c3", "d7d6",
            ],
        ),
        _seq(
            "italian_developed_castled",
            ("opening", "open", "developed", "castled"),
            [
                "e2e4", "e7e5",
                "g1f3", "b8c6",
                "f1c4", "g8f6",
                "d2d3", "f8c5",
                "c2c3", "d7d6",
                "e1g1", "e8g8",
            ],
        ),
        _seq(
            "ruy_lopez_closed_development",
            ("opening", "open", "developed", "castled"),
            [
                "e2e4", "e7e5",
                "g1f3", "b8c6",
                "f1b5", "a7a6",
                "b5a4", "g8f6",
                "e1g1", "f8e7",
                "f1e1", "b7b5",
                "a4b3", "d7d6",
                "c2c3", "e8g8",
            ],
        ),
        _seq(
            "scotch_open_center",
            ("opening", "open", "center"),
            [
                "e2e4", "e7e5",
                "g1f3", "b8c6",
                "d2d4", "e5d4",
                "f3d4", "g8f6",
                "b1c3", "f8b4",
                "d4c6", "b7c6",
            ],
        ),
        _seq(
            "open_sicilian_development",
            ("opening", "open", "imbalanced"),
            [
                "e2e4", "c7c5",
                "g1f3", "d7d6",
                "d2d4", "c5d4",
                "f3d4", "g8f6",
                "b1c3", "a7a6",
                "c1e3", "e7e5",
            ],
        ),
        _seq(
            "sicilian_dragon_like",
            ("opening", "open", "imbalanced", "kingside_fianchetto"),
            [
                "e2e4", "c7c5",
                "g1f3", "d7d6",
                "d2d4", "c5d4",
                "f3d4", "g8f6",
                "b1c3", "g7g6",
                "c1e3", "f8g7",
                "f2f3", "e8g8",
                "d1d2",
            ],
        ),
        _seq(
            "queens_gambit_declined",
            ("opening", "closed", "developed"),
            [
                "d2d4", "d7d5",
                "c2c4", "e7e6",
                "b1c3", "g8f6",
                "c1g5", "f8e7",
                "e2e3", "e8g8",
                "g1f3", "h7h6",
                "g5h4", "b7b6",
            ],
        ),
        _seq(
            "slav_structure",
            ("opening", "semi_closed", "structure"),
            [
                "d2d4", "d7d5",
                "c2c4", "c7c6",
                "g1f3", "g8f6",
                "b1c3", "d5c4",
                "a2a4", "c8f5",
                "e2e3", "e7e6",
            ],
        ),
        _seq(
            "caro_kann_advance",
            ("opening", "semi_closed", "structure"),
            [
                "e2e4", "c7c6",
                "d2d4", "d7d5",
                "e4e5", "c8f5",
                "g1f3", "e7e6",
                "f1e2", "c6c5",
                "c2c3", "b8c6",
            ],
        ),
        _seq(
            "french_advance_closed",
            ("opening", "closed", "locked_center"),
            [
                "e2e4", "e7e6",
                "d2d4", "d7d5",
                "e4e5", "c7c5",
                "c2c3", "b8c6",
                "g1f3", "d8b6",
                "a2a3", "c5c4",
            ],
        ),

        # ------------------------------------------------------------------
        # Middlegames
        # ------------------------------------------------------------------
        _seq(
            "kings_indian_closed_center",
            ("middlegame", "closed", "locked_center", "kingside_attack"),
            [
                "d2d4", "g8f6",
                "c2c4", "g7g6",
                "b1c3", "f8g7",
                "e2e4", "d7d6",
                "g1f3", "e8g8",
                "f1e2", "e7e5",
                "d4d5", "a7a5",
                "e1g1", "b8a6",
            ],
        ),
        _seq(
            "open_center_castled",
            ("middlegame", "open", "castled"),
            [
                "e2e4", "e7e5",
                "g1f3", "b8c6",
                "f1b5", "a7a6",
                "b5a4", "g8f6",
                "e1g1", "f8e7",
                "f1e1", "b7b5",
                "a4b3", "d7d6",
                "c2c3", "e8g8",
                "d2d4",
            ],
        ),
        _seq(
            "queenless_middlegame",
            ("middlegame", "queenless", "endgame_like"),
            [
                "e2e4", "e7e5",
                "g1f3", "b8c6",
                "f1b5", "a7a6",
                "b5c6", "d7c6",
                "e1g1", "f7f6",
                "d2d4", "e5d4",
                "f3d4", "c6c5",
                "d4f5", "d8d1",
                "f1d1",
            ],
        ),
        _seq(
            "isolated_queen_pawn",
            ("middlegame", "open", "isolated_pawn"),
            [
                "d2d4", "d7d5",
                "c2c4", "e7e6",
                "b1c3", "g8f6",
                "c4d5", "e6d5",
                "c1g5", "f8e7",
                "e2e3", "e8g8",
                "f1d3", "c7c6",
                "g1f3",
            ],
        ),
        _seq(
            "hanging_pawns_structure",
            ("middlegame", "semi_open", "structure"),
            [
                "d2d4", "d7d5",
                "c2c4", "e7e6",
                "g1f3", "g8f6",
                "b1c3", "f8e7",
                "c4d5", "e6d5",
                "c1g5", "e8g8",
                "e2e3", "c7c5",
                "f1d3", "c5c4",
            ],
        ),
        _fen(
            "opposite_side_castling_attack",
            ("middlegame", "attack", "opposite_castling", "king_safety"),
            "2kr1bnr/pppq1ppp/2np4/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQ - 0 8",
        ),
        _fen(
            "exposed_white_king_middlegame",
            ("middlegame", "king_safety", "exposed_king"),
            "r1bq1rk1/ppppbppp/2n2n2/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQ - 0 8",
        ),
        _fen(
            "open_file_near_black_king",
            ("middlegame", "king_safety", "open_file"),
            "r4rk1/ppp2ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R2QR1K1 w - - 0 10",
        ),

        # ------------------------------------------------------------------
        # Tactical / attacking starts
        # ------------------------------------------------------------------
        _fen(
            "white_attack_on_king",
            ("middlegame", "attack", "king_safety"),
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 9",
        ),
        _fen(
            "black_attack_on_king",
            ("middlegame", "attack", "king_safety"),
            "r1bqr1k1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 0 9",
        ),
        _fen(
            "material_imbalance_minor_for_pawns",
            ("middlegame", "imbalance", "material"),
            "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP4/PPP2PPP/R1BQ1RK1 w - - 0 9",
        ),

        # ------------------------------------------------------------------
        # Endgames: simple, complex, won, drawn-ish, and mating material.
        # ------------------------------------------------------------------
        _fen(
            "king_pawn_endgame_opposition",
            ("endgame", "pawn", "balanced"),
            "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1",
        ),
        _fen(
            "king_pawn_race",
            ("endgame", "pawn", "race"),
            "8/2k5/8/3P4/6p1/8/6K1/8 w - - 0 1",
        ),
        _fen(
            "rook_endgame_equal_rooks",
            ("endgame", "rook", "balanced", "drawish"),
            "6k1/8/8/8/8/8/6K1/R6r w - - 0 1",
        ),
        _fen(
            "rook_endgame_lucena_like",
            ("endgame", "rook", "winning_technique"),
            "4k3/4P3/4K3/8/8/8/8/R6r w - - 0 1",
        ),
        _fen(
            "rook_endgame_philidor_like",
            ("endgame", "rook", "drawish", "defensive_technique"),
            "4k3/8/4K3/4P3/8/8/8/R6r b - - 0 1",
        ),
        _fen(
            "queen_and_pawn_vs_king",
            ("endgame", "queen", "pawn", "winning"),
            "6k1/6P1/8/8/8/8/8/6KQ w - - 0 1",
        ),
        _fen(
            "queen_vs_rook_endgame",
            ("endgame", "queen", "rook", "winning"),
            "6k1/8/8/8/8/8/5r2/5QK1 w - - 0 1",
        ),
        _fen(
            "double_queen_vs_king",
            ("endgame", "double_queen", "winning", "mating_material"),
            "6k1/8/8/8/8/8/8/QQ4K1 w - - 0 1",
        ),
        _fen(
            "two_bishop_checkmate_material",
            ("endgame", "two_bishops", "mating_material", "winning_technique"),
            "7k/8/8/8/8/2B5/3B4/6K1 w - - 0 1",
        ),
        _fen(
            "bishop_knight_checkmate_material",
            ("endgame", "bishop_knight", "mating_material", "winning_technique"),
            "7k/8/8/8/8/2B5/3N4/6K1 w - - 0 1",
        ),
        _fen(
            "opposite_colored_bishops_drawish",
            ("endgame", "bishop", "opposite_colored_bishops", "drawish", "balanced"),
            "6k1/8/8/8/8/2B5/6K1/5b2 w - - 0 1",
        ),
        _fen(
            "same_colored_bishops_balanced",
            ("endgame", "bishop", "balanced", "drawish"),
            "6k1/8/8/8/8/2B5/6K1/2b5 w - - 0 1",
        ),
        _fen(
            "knight_vs_bishop_balanced",
            ("endgame", "minor_piece", "balanced", "drawish"),
            "6k1/8/8/8/8/2N5/6K1/5b2 w - - 0 1",
        ),
        _fen(
            "queen_vs_pawn_near_promotion",
            ("endgame", "queen", "pawn", "defensive_resource"),
            "6k1/6p1/8/8/8/8/8/6KQ b - - 0 1",
        ),
        _fen(
            "king_rook_vs_king",
            ("endgame", "rook", "mating_material", "winning_technique"),
            "7k/8/8/8/8/8/8/R5K1 w - - 0 1",
        ),
    ]

    return positions


def load_positions_json(path: str) -> list[StartPosition]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    positions: list[StartPosition] = []

    for i, item in enumerate(payload):
        label = item.get("label", f"pos_{i:04d}")
        fen = item["fen"]
        tags = tuple(item.get("tags", []))

        positions.append(StartPosition(label=label, fen=fen, tags=tags))

    return positions


def generate_start_positions(
    *,
    source: str,
    positions_json: str | None,
    start_fen: str,
    games: int,
    random_openings: int,
    seed: int,
) -> list[StartPosition]:
    rng = random.Random(seed)

    if positions_json is not None:
        pool = load_positions_json(positions_json)
    elif source == "suite":
        pool = built_in_position_pool()
    elif source == "random":
        pool = [
            StartPosition(
                label=f"random_{i:04d}",
                fen=make_random_opening_fen(
                    start_fen=start_fen,
                    random_plies=random_openings,
                    rng=rng,
                ),
                tags=("random",),
            )
            for i in range(games)
        ]
    else:
        raise ValueError(f"unknown position source: {source!r}")

    if not pool:
        raise ValueError("position pool is empty")

    selected: list[StartPosition] = []

    while len(selected) < games:
        chunk = pool[:]
        rng.shuffle(chunk)
        selected.extend(chunk)

    return selected[:games]


def score_for_label(result: GameResult, label: str) -> float:
    if result.winner == "draw":
        return 0.5

    if result.winner == "white":
        return 1.0 if result.white_label == label else 0.0

    if result.winner == "black":
        return 1.0 if result.black_label == label else 0.0

    raise AssertionError(result.winner)


def update_ordered_stats(
    stats: OrderedStats,
    *,
    result: GameResult,
    label: str,
) -> None:
    s = score_for_label(result, label)

    stats.games += 1
    stats.score_sum += s
    stats.plies += result.plies
    stats.reasons[result.reason] += 1

    if s == 1.0:
        stats.wins += 1
    elif s == 0.5:
        stats.draws += 1
    else:
        stats.losses += 1


def chunk_indexed_positions(
    start_positions: list[StartPosition],
    *,
    chunk_size: int,
) -> list[list[tuple[int, StartPosition]]]:
    indexed = list(enumerate(start_positions))

    return [
        indexed[i : i + chunk_size]
        for i in range(0, len(indexed), chunk_size)
    ]


def _play_matchup_chunk_worker(
    *,
    a: ModelSpec,
    b: ModelSpec,
    indexed_positions: list[tuple[int, StartPosition]],
    max_plies: int,
    seed: int,
    game_idx_start: int,
) -> list[GameResult]:
    """
    Worker process function.

    Each worker creates its own agent instances and plays a chunk of positions.
    This avoids sharing mutable Board/Agent state across processes.
    """
    results: list[GameResult] = []

    # Separate agent instances for color directions.
    a_agent_as_white = make_agent(a, seed=seed + 101)
    b_agent_as_black = make_agent(b, seed=seed + 202)

    b_agent_as_white = make_agent(b, seed=seed + 303)
    a_agent_as_black = make_agent(a, seed=seed + 404)

    for pos_idx, start in indexed_positions:
        # Two games per start position.
        # Derive stable unique IDs from global position index.
        game_idx_1 = game_idx_start + 2 * pos_idx
        game_idx_2 = game_idx_start + 2 * pos_idx + 1

        r1 = play_game(
            game_idx=game_idx_1,
            start_fen=start.fen,
            white_agent=a_agent_as_white,
            black_agent=b_agent_as_black,
            white_label=a.label,
            black_label=b.label,
            max_plies=max_plies,
        )
        results.append(r1)

        r2 = play_game(
            game_idx=game_idx_2,
            start_fen=start.fen,
            white_agent=b_agent_as_white,
            black_agent=a_agent_as_black,
            white_label=b.label,
            black_label=a.label,
            max_plies=max_plies,
        )
        results.append(r2)

    return results


def play_matchup(
    *,
    a: ModelSpec,
    b: ModelSpec,
    start_positions: list[StartPosition],
    max_plies: int,
    seed: int,
    game_idx_start: int,
    use_progress: bool,
    progress_every: int,
    workers: int,
    chunk_size: int | None,
) -> tuple[list[GameResult], int]:
    if not start_positions:
        return [], game_idx_start

    workers = max(1, workers)

    # Sequential path: keeps old behavior and avoids multiprocessing overhead.
    if workers == 1:
        results: list[GameResult] = []
        game_idx = game_idx_start

        a_agent_as_white = make_agent(a, seed=seed + 101)
        b_agent_as_black = make_agent(b, seed=seed + 202)

        b_agent_as_white = make_agent(b, seed=seed + 303)
        a_agent_as_black = make_agent(a, seed=seed + 404)

        if use_progress and tqdm is not None:
            iterator = tqdm(
                start_positions,
                total=len(start_positions),
                desc=f"{a.label} vs {b.label}",
                unit="pos",
                dynamic_ncols=True,
                mininterval=0.5,
            )
        else:
            iterator = start_positions

        t0 = time.time()

        for i, start in enumerate(iterator, start=1):
            r1 = play_game(
                game_idx=game_idx,
                start_fen=start.fen,
                white_agent=a_agent_as_white,
                black_agent=b_agent_as_black,
                white_label=a.label,
                black_label=b.label,
                max_plies=max_plies,
            )
            results.append(r1)
            game_idx += 1

            r2 = play_game(
                game_idx=game_idx,
                start_fen=start.fen,
                white_agent=b_agent_as_white,
                black_agent=a_agent_as_black,
                white_label=b.label,
                black_label=a.label,
                max_plies=max_plies,
            )
            results.append(r2)
            game_idx += 1

            if use_progress and tqdm is not None:
                reason_counts = Counter(r.reason for r in results)
                iterator.set_postfix(
                    games=len(results),
                    mate=reason_counts.get("checkmate", 0),
                    draw=sum(1 for r in results if r.winner == "draw"),
                    rep=reason_counts.get("threefold repetition", 0),
                    max=reason_counts.get("max plies", 0),
                )
            elif progress_every > 0 and i % progress_every == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-9)
                remaining = len(start_positions) - i
                eta = remaining / max(rate, 1e-9)

                print(
                    f"  {a.label} vs {b.label}: "
                    f"{i}/{len(start_positions)} positions "
                    f"({2 * i} games), "
                    f"elapsed={elapsed:.1f}s, "
                    f"eta={eta:.1f}s"
                )

        return results, game_idx

    # Parallel path.
    total_positions = len(start_positions)

    if chunk_size is None or chunk_size <= 0:
        # More chunks than workers gives better load balancing.
        target_chunks = workers * 4
        chunk_size = max(1, (total_positions + target_chunks - 1) // target_chunks)

    chunks = chunk_indexed_positions(
        start_positions,
        chunk_size=chunk_size,
    )

    results: list[GameResult] = []

    t0 = time.time()
    completed_positions = 0

    print(
        f"parallel matchup: {a.label} vs {b.label} | "
        f"workers={workers}, chunks={len(chunks)}, chunk_size={chunk_size}"
    )

    progress_bar = None
    if use_progress and tqdm is not None:
        progress_bar = tqdm(
            total=total_positions,
            desc=f"{a.label} vs {b.label}",
            unit="pos",
            dynamic_ncols=True,
            mininterval=0.5,
        )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_chunk: dict[Any, list[tuple[int, StartPosition]]] = {}

        for chunk_idx, chunk in enumerate(chunks):
            fut = pool.submit(
                _play_matchup_chunk_worker,
                a=a,
                b=b,
                indexed_positions=chunk,
                max_plies=max_plies,
                seed=seed + 1000 * chunk_idx,
                game_idx_start=game_idx_start,
            )
            future_to_chunk[fut] = chunk

        for fut in as_completed(future_to_chunk):
            chunk = future_to_chunk[fut]
            chunk_results = fut.result()

            results.extend(chunk_results)
            completed_positions += len(chunk)

            if progress_bar is not None:
                reason_counts = Counter(r.reason for r in results)
                progress_bar.update(len(chunk))
                progress_bar.set_postfix(
                    games=len(results),
                    mate=reason_counts.get("checkmate", 0),
                    draw=sum(1 for r in results if r.winner == "draw"),
                    rep=reason_counts.get("threefold repetition", 0),
                    max=reason_counts.get("max plies", 0),
                )
            elif progress_every > 0:
                elapsed = time.time() - t0
                rate = completed_positions / max(elapsed, 1e-9)
                remaining = total_positions - completed_positions
                eta = remaining / max(rate, 1e-9)

                print(
                    f"  {a.label} vs {b.label}: "
                    f"{completed_positions}/{total_positions} positions "
                    f"({len(results)} games), "
                    f"elapsed={elapsed:.1f}s, "
                    f"eta={eta:.1f}s"
                )

    if progress_bar is not None:
        progress_bar.close()

    # Parallel futures complete out of order. Sort back into deterministic order.
    results.sort(key=lambda r: r.game_idx)

    next_game_idx = game_idx_start + 2 * total_positions
    return results, next_game_idx


def print_models(models: list[ModelSpec]) -> None:
    print()
    print("=" * 100)
    print("MODELS")
    print("=" * 100)

    for m in models:
        print(f"{m.label}:")
        print(f"  kind: {m.kind}")
        print(f"  path: {m.path}")

        if m.kind == "search":
            print(f"  search_depth: {m.search_depth}")
            print(f"  search_max_branch: {m.search_max_branch}")
            print(f"  search_draw_safety: {m.search_use_draw_safety}")
            print(f"  search_tactical_safety: {m.search_use_tactical_safety}")

        if m.kind in ("search", "material_search"):
            print(f"  search_depth: {m.search_depth}")

def print_position_summary(positions: list[StartPosition]) -> None:
    tag_counts: Counter[str] = Counter()

    for pos in positions:
        for tag in pos.tags:
            tag_counts[tag] += 1

    print()
    print("=" * 100)
    print("POSITIONS")
    print("=" * 100)
    print(f"positions used: {len(positions)}")

    if tag_counts:
        print()
        print("Position tags:")
        for tag, count in tag_counts.most_common():
            print(f"  {tag}: {count}")

    print()
    print("Selected positions:")
    for i, pos in enumerate(positions, start=1):
        tags = ",".join(pos.tags)
        print(f"  {i:3d}. {pos.label:35s} [{tags}]")


def print_pair_summary(
    *,
    a: ModelSpec,
    b: ModelSpec,
    stats: dict[tuple[str, str], OrderedStats],
) -> None:
    ab = stats[(a.label, b.label)]
    ba = stats[(b.label, a.label)]

    print()
    print("-" * 100)
    print(f"{a.label} vs {b.label}")
    print("-" * 100)

    print(
        f"{a.label}: "
        f"{ab.wins}W-{ab.draws}D-{ab.losses}L "
        f"score={ab.score_rate:.1%} "
        f"avg_plies={ab.avg_plies:.1f}"
    )

    print(
        f"{b.label}: "
        f"{ba.wins}W-{ba.draws}D-{ba.losses}L "
        f"score={ba.score_rate:.1%} "
        f"avg_plies={ba.avg_plies:.1f}"
    )

    # ab.reasons and ba.reasons contain the same games from opposite perspectives,
    # so do not add them together.
    print("Reasons:")
    for reason, count in ab.reasons.most_common():
        print(f"  {reason}: {count}")


def print_score_matrix(
    *,
    models: list[ModelSpec],
    stats: dict[tuple[str, str], OrderedStats],
) -> None:
    labels = [m.label for m in models]

    print()
    print("=" * 100)
    print("SCORE MATRIX")
    print("=" * 100)
    print("Rows score against columns. 50% means equal.")

    width = max(10, max(len(x) for x in labels) + 2)

    header = " " * width
    for label in labels:
        header += f"{label:>{width}s}"
    print(header)

    for row in labels:
        line = f"{row:>{width}s}"

        for col in labels:
            if row == col and (row, col) not in stats:
                cell = "—"
            else:
                s = stats.get((row, col))
                if s is None or s.games == 0:
                    cell = "—"
                else:
                    cell = f"{100.0 * s.score_rate:.1f}%"

            line += f"{cell:>{width}s}"

        print(line)

def print_wdl_matrix(
    *,
    models: list[ModelSpec],
    stats: dict[tuple[str, str], OrderedStats],
) -> None:
    labels = [m.label for m in models]

    print()
    print("=" * 100)
    print("W-D-L MATRIX")
    print("=" * 100)
    print("Rows are scored from the row model's perspective.")

    width = max(10, max(len(x) for x in labels) + 2)

    header = " " * width
    for label in labels:
        header += f"{label:>{width}s}"
    print(header)

    for row in labels:
        line = f"{row:>{width}s}"

        for col in labels:
            if row == col:
                cell = "—"
            else:
                s = stats.get((row, col))
                if s is None or s.games == 0:
                    cell = "—"
                else:
                    cell = f"{s.wins}-{s.draws}-{s.losses}"

            line += f"{cell:>{width}s}"

        print(line)


def print_avg_plies_matrix(
    *,
    models: list[ModelSpec],
    stats: dict[tuple[str, str], OrderedStats],
) -> None:
    labels = [m.label for m in models]

    print()
    print("=" * 100)
    print("AVG PLIES MATRIX")
    print("=" * 100)
    print("Average game length for each ordered matchup.")

    width = max(10, max(len(x) for x in labels) + 2)

    header = " " * width
    for label in labels:
        header += f"{label:>{width}s}"
    print(header)

    for row in labels:
        line = f"{row:>{width}s}"

        for col in labels:
            if row == col:
                cell = "—"
            else:
                s = stats.get((row, col))
                if s is None or s.games == 0:
                    cell = "—"
                else:
                    cell = f"{s.avg_plies:.1f}"

            line += f"{cell:>{width}s}"

        print(line)

def print_detailed_table(
    *,
    models: list[ModelSpec],
    stats: dict[tuple[str, str], OrderedStats],
) -> None:
    print()
    print("=" * 100)
    print("DETAILED ORDERED RESULTS")
    print("=" * 100)

    labels = [m.label for m in models]

    for row in labels:
        for col in labels:
            if row == col and (row, col) not in stats:
                continue

            s = stats.get((row, col))
            if s is None or s.games == 0:
                continue

            print(
                f"{row:20s} vs {col:20s} "
                f"{s.wins:4d}W-{s.draws:4d}D-{s.losses:4d}L "
                f"score={s.score_rate:7.2%} "
                f"games={s.games:4d} "
                f"avg_plies={s.avg_plies:6.1f}"
            )


def print_tag_summary(
    *,
    models: list[ModelSpec],
    tag_stats: dict[tuple[str, str, str], OrderedStats],
) -> None:
    labels = [m.label for m in models]
    tags = sorted({tag for (_a, _b, tag) in tag_stats.keys()})

    if not tags:
        return

    print()
    print("=" * 100)
    print("TAG / CATEGORY SUMMARY")
    print("=" * 100)
    print("Rows score against columns within each position tag.")

    for tag in tags:
        print()
        print(f"[{tag}]")

        width = max(10, max(len(x) for x in labels) + 2)

        header = " " * width
        for label in labels:
            header += f"{label:>{width}s}"
        print(header)

        for row in labels:
            line = f"{row:>{width}s}"

            for col in labels:
                s = tag_stats.get((row, col, tag))
                if s is None or s.games == 0:
                    cell = "—"
                else:
                    cell = f"{100.0 * s.score_rate:.1f}%"

                line += f"{cell:>{width}s}"

            print(line)


def save_json(
    *,
    path: str,
    models: list[ModelSpec],
    start_positions: list[StartPosition],
    results: list[GameResult],
    stats: dict[tuple[str, str], OrderedStats],
    tag_stats: dict[tuple[str, str, str], OrderedStats],
    behavior_stats: dict[str, OpeningBehaviorStats],
) -> None:
    payload: dict[str, Any] = {
        "models": [asdict(m) for m in models],
        "start_positions": [asdict(p) for p in start_positions],
        "results": [asdict(r) for r in results],
        "stats": {
            f"{a}__vs__{b}": {
                "games": s.games,
                "wins": s.wins,
                "draws": s.draws,
                "losses": s.losses,
                "score": s.score_rate,
                "avg_plies": s.avg_plies,
                "reasons": dict(s.reasons),
            }
            for (a, b), s in stats.items()
        },
        "tag_stats": {
            f"{a}__vs__{b}__tag__{tag}": {
                "games": s.games,
                "wins": s.wins,
                "draws": s.draws,
                "losses": s.losses,
                "score": s.score_rate,
                "avg_plies": s.avg_plies,
                "reasons": dict(s.reasons),
            }
            for (a, b, tag), s in tag_stats.items()
        },
        "opening_behavior_stats": {
            label: {
                "games": s.games,
                "avg_queen_moves_10": s.avg_queen_moves_10,
                "queen_out_rate_10": s.queen_out_rate_10,
                "castle_rate_20": s.castle_rate_20,
                "avg_minor_dev_10": s.avg_minor_dev_10,
                "avg_center_pawns_8": s.avg_center_pawns_8,
                "raw": {
                    "queen_moves_10": s.queen_moves_10,
                    "queen_out_10": s.queen_out_10,
                    "castled_20": s.castled_20,
                    "minor_dev_10": s.minor_dev_10,
                    "center_pawns_8": s.center_pawns_8,
                },
            }
            for label, s in behavior_stats.items()
        },
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {out}")

def save_compact_json(
    *,
    path: str,
    models: list[ModelSpec],
    start_positions: list[StartPosition],
    stats: dict[tuple[str, str], OrderedStats],
) -> None:
    labels = [m.label for m in models]

    def stat_payload(s: OrderedStats | None) -> dict[str, Any] | None:
        if s is None or s.games == 0:
            return None

        return {
            "games": s.games,
            "wins": s.wins,
            "draws": s.draws,
            "losses": s.losses,
            "wdl": f"{s.wins}-{s.draws}-{s.losses}",
            "score": s.score_rate,
            "avg_plies": s.avg_plies,
            "reasons": dict(s.reasons),
        }

    score_matrix: dict[str, dict[str, float | None]] = {}
    wdl_matrix: dict[str, dict[str, str | None]] = {}
    avg_plies_matrix: dict[str, dict[str, float | None]] = {}

    for row in labels:
        score_matrix[row] = {}
        wdl_matrix[row] = {}
        avg_plies_matrix[row] = {}

        for col in labels:
            if row == col:
                score_matrix[row][col] = None
                wdl_matrix[row][col] = None
                avg_plies_matrix[row][col] = None
                continue

            s = stats.get((row, col))

            if s is None or s.games == 0:
                score_matrix[row][col] = None
                wdl_matrix[row][col] = None
                avg_plies_matrix[row][col] = None
            else:
                score_matrix[row][col] = s.score_rate
                wdl_matrix[row][col] = f"{s.wins}-{s.draws}-{s.losses}"
                avg_plies_matrix[row][col] = s.avg_plies

    payload: dict[str, Any] = {
        "models": [asdict(m) for m in models],
        "position_count": len(start_positions),
        "stats": {
            f"{a}__vs__{b}": stat_payload(s)
            for (a, b), s in stats.items()
        },
        "score_matrix": score_matrix,
        "wdl_matrix": wdl_matrix,
        "avg_plies_matrix": avg_plies_matrix,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved compact JSON: {out}")

def update_opening_behavior_stats(
    stats: OpeningBehaviorStats,
    *,
    result: GameResult,
    label: str,
) -> None:
    stats.games += 1

    if result.white_label == label:
        stats.queen_moves_10 += result.white_queen_moves_10
        stats.queen_out_10 += int(result.white_queen_out_10)
        stats.castled_20 += int(result.white_castled_20)
        stats.minor_dev_10 += result.white_minor_dev_10
        stats.center_pawns_8 += result.white_center_pawns_8
        return

    if result.black_label == label:
        stats.queen_moves_10 += result.black_queen_moves_10
        stats.queen_out_10 += int(result.black_queen_out_10)
        stats.castled_20 += int(result.black_castled_20)
        stats.minor_dev_10 += result.black_minor_dev_10
        stats.center_pawns_8 += result.black_center_pawns_8
        return

    raise ValueError(f"label {label!r} not found in result {result}")

def print_opening_behavior_summary(
    *,
    models: list[ModelSpec],
    behavior_stats: dict[str, OpeningBehaviorStats],
) -> None:
    print()
    print("=" * 100)
    print("OPENING / STYLE BEHAVIOR SUMMARY")
    print("=" * 100)
    print("Per model, aggregated over all games in this evaluation.")
    print("Q<=10 = average queen moves in first 10 plies.")
    print("Qout% = games where the queen moved in first 10 plies.")
    print("Castle% = games where the side castled by ply 20.")
    print("MinorDev = average developed bishops/knights by ply 10.")
    print("CtrPawns = average pawns occupying d4/e4/d5/e5 by ply 8.")
    print()

    print(
        f"{'model':24s} "
        f"{'games':>6s} "
        f"{'Q<=10':>8s} "
        f"{'Qout%':>8s} "
        f"{'Castle%':>9s} "
        f"{'MinorDev':>9s} "
        f"{'CtrPawns':>9s}"
    )

    for m in models:
        s = behavior_stats.get(m.label)
        if s is None or s.games == 0:
            continue

        print(
            f"{m.label:24s} "
            f"{s.games:6d} "
            f"{s.avg_queen_moves_10:8.2f} "
            f"{100.0 * s.queen_out_rate_10:7.1f}% "
            f"{100.0 * s.castle_rate_20:8.1f}% "
            f"{s.avg_minor_dev_10:9.2f} "
            f"{s.avg_center_pawns_8:9.2f}"
        )

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model spec. Format: label:kind:path or "
            "label:kind:path:depth=2,max_branch=8,draw=1,tactical=1. "
            "Kinds: plain, tactical, search, material, random."
        ),
    )

    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument(
        "--start-fen",
        default=START_FEN,
    )

    ap.add_argument(
        "--position-source",
        choices=["suite", "random"],
        default="suite",
        help="Start-position source. 'suite' uses curated positions; 'random' uses random legal plies.",
    )

    ap.add_argument(
        "--positions-json",
        default=None,
        help="Optional JSON file of FEN start positions. Overrides --position-source.",
    )

    ap.add_argument(
        "--random-openings",
        type=int,
        default=6,
        help="Only used when --position-source random. Number of random legal plies before each game.",
    )

    ap.add_argument(
        "--include-self",
        action="store_true",
        help="Also evaluate each model against itself.",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for parallel game evaluation. "
            "Use 1 for sequential. For your Ryzen 5 5600, 6-11 are reasonable values."
        ),
    )

    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Positions per worker task. Smaller improves load balancing; "
            "larger reduces checkpoint-loading overhead. Try 2 first."
        ),
    )

    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    ap.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Fallback text progress interval when tqdm is disabled/unavailable.",
    )

    ap.add_argument(
        "--compact-report",
        action="store_true",
        help=(
            "Print only compact report output: pair summaries, score matrix, "
            "W-D-L matrix, avg plies matrix, and detailed ordered results. "
            "Suppresses position listing, opening behavior, and tag summaries."
        ),
    )

    ap.add_argument(
        "--compact-json",
        action="store_true",
        help="Write compact JSON without per-game results or tag tables.",
    )

    ap.add_argument(
        "--pair-mode",
        choices=["all", "adjacent"],
        default="all",
        help=(
            "all = round-robin all unordered pairs. "
            "adjacent = only model[i] vs model[i+1], useful for model ladders."
        ),
    )

    ap.add_argument("--json-out", default=None)

    args = ap.parse_args()

    models = [parse_model(x) for x in args.model]

    labels = [m.label for m in models]
    if len(set(labels)) != len(labels):
        raise ValueError("model labels must be unique")

    behavior_stats: dict[str, OpeningBehaviorStats] = defaultdict(OpeningBehaviorStats)

    print_models(models)

    print()
    print("=" * 100)
    print("START POSITION CONFIG")
    print("=" * 100)
    print(f"games per unordered matchup:        {args.games}")
    print(f"actual games per unordered matchup: {args.games * 2}")
    print(f"position source:                    {args.position_source}")
    print(f"positions json:                     {args.positions_json}")
    print(f"random opening plies:               {args.random_openings}")
    print(f"workers:                            {args.workers}")
    print(f"chunk size:                         {args.chunk_size}")
    print(f"seed:                               {args.seed}")

    start_positions = generate_start_positions(
        source=args.position_source,
        positions_json=args.positions_json,
        start_fen=args.start_fen,
        games=args.games,
        random_openings=args.random_openings,
        seed=args.seed,
    )

    if args.compact_report:
        print()
        print("=" * 100)
        print("POSITIONS")
        print("=" * 100)
        print(f"positions used: {len(start_positions)}")
    else:
        print_position_summary(start_positions)

    stats: dict[tuple[str, str], OrderedStats] = defaultdict(OrderedStats)
    tag_stats: dict[tuple[str, str, str], OrderedStats] = defaultdict(OrderedStats)

    all_results: list[GameResult] = []

    if args.pair_mode == "adjacent":
        if args.include_self:
            raise ValueError("--include-self does not make sense with --pair-mode adjacent")

        if len(models) < 2:
            raise ValueError("--pair-mode adjacent needs at least two models")

        pairs = list(zip(models[:-1], models[1:]))

    else:
        if args.include_self:
            pairs = list(itertools.combinations_with_replacement(models, 2))
        else:
            pairs = list(itertools.combinations(models, 2))

    t0 = time.time()
    game_idx = 1

    # Map FEN -> tags for category summaries.
    # If the same FEN appears multiple times with different labels/tags, tags are merged.
    fen_tags: dict[str, set[str]] = defaultdict(set)
    for pos in start_positions:
        fen_tags[pos.fen].update(pos.tags)

    for pair_idx, (a, b) in enumerate(pairs, start=1):
        print()
        print("=" * 100)
        print(f"MATCHUP {pair_idx}/{len(pairs)}: {a.label} vs {b.label}")
        print("=" * 100)

        pair_seed = args.seed + 10_000 * pair_idx

        results, game_idx = play_matchup(
            a=a,
            b=b,
            start_positions=start_positions,
            max_plies=args.max_plies,
            seed=pair_seed,
            game_idx_start=game_idx,
            use_progress=not args.no_progress,
            progress_every=args.progress_every,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )

        all_results.extend(results)

        for r in results:
            update_ordered_stats(
                stats[(r.white_label, r.black_label)],
                result=r,
                label=r.white_label,
            )
            update_ordered_stats(
                stats[(r.black_label, r.white_label)],
                result=r,
                label=r.black_label,
            )

            tags = fen_tags.get(r.start_fen, set())
            for tag in tags:
                update_ordered_stats(
                    tag_stats[(r.white_label, r.black_label, tag)],
                    result=r,
                    label=r.white_label,
                )
                update_ordered_stats(
                    tag_stats[(r.black_label, r.white_label, tag)],
                    result=r,
                    label=r.black_label,
                )

            update_opening_behavior_stats(
                behavior_stats[r.white_label],
                result=r,
                label=r.white_label,
            )

            update_opening_behavior_stats(
                behavior_stats[r.black_label],
                result=r,
                label=r.black_label,
            )

        print_pair_summary(a=a, b=b, stats=stats)

    dt = time.time() - t0

    print_score_matrix(models=models, stats=stats)
    print_wdl_matrix(models=models, stats=stats)
    print_avg_plies_matrix(models=models, stats=stats)
    print_detailed_table(models=models, stats=stats)

    if not args.compact_report:
        print_opening_behavior_summary(
            models=models,
            behavior_stats=behavior_stats,
        )

        print_tag_summary(models=models, tag_stats=tag_stats)

    print()
    print("=" * 100)
    print("RUNTIME")
    print("=" * 100)
    print(f"total games: {len(all_results)}")
    print(f"wall time:   {dt:.2f}s")
    print(f"games/sec:   {len(all_results) / max(dt, 1e-9):.2f}")

    if args.json_out:
        if args.compact_json or args.compact_report:
            save_compact_json(
                path=args.json_out,
                models=models,
                start_positions=start_positions,
                stats=stats,
            )
        else:
            save_json(
                path=args.json_out,
                models=models,
                start_positions=start_positions,
                results=all_results,
                stats=stats,
                tag_stats=tag_stats,
                behavior_stats=behavior_stats,
            )


if __name__ == "__main__":
    main()