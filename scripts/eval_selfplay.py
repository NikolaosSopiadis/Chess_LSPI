from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Literal

from chess_core.board import Board
from chess_core.move import Move, F_CASTLE
from chess_core.piece import Piece as p
from chess_rl.rewards.v1_terminal_plus_potential import material_potential
from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.material_greedy import MaterialGreedyAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent
from chess_rl.agents.lspi_tactical import LSPITacticalAgent
from chess_rl.agents.lspi_search import LSPISearchAgent
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

Winner = Literal["white", "black", "draw"]


@dataclass(frozen=True, slots=True)
class GameResult:
    game_idx: int
    white_label: str
    black_label: str
    winner: Winner
    reason: str
    plies: int
    start_fen: str
    final_fen: str
    illegal_move: str | None = None

    # Opening/style diagnostics.
    white_queen_moves_10: int = 0
    black_queen_moves_10: int = 0

    white_queen_out_10: bool = False
    black_queen_out_10: bool = False

    white_castled_20: bool = False
    black_castled_20: bool = False

    white_minor_dev_10: int = 0
    black_minor_dev_10: int = 0

    white_center_pawns_8: int = 0
    black_center_pawns_8: int = 0

@dataclass
class AgentScore:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games


def _move_to_uci_like(board: Board, move: Move) -> str:
    # Mostly for diagnostics. Promotion suffix is numeric here unless you expand it.
    s = board.idx_to_algebraic(move.src_square) + board.idx_to_algebraic(move.dst_square)
    if move.promotion:
        s += f"={move.promotion}"
    return s


def _canonical_kind(kind: str) -> str:
    k = kind.lower().strip()

    if k == "random":
        return "random"

    if k in ("material", "material_greedy", "greedy_material"):
        return "material"

    if k in ("lspi_plain", "plain", "lspi_v1"):
        return "lspi_plain"

    if k in ("lspi_tactical", "tactical", "safe", "lspi_safe"):
        return "lspi_tactical"

    if k in ("lspi_search", "search"):
        return "lspi_search"

    raise ValueError(f"unknown agent kind: {kind!r}")


def _agent_label(kind: str, path: str | None) -> str:
    canonical = _canonical_kind(kind)

    if path:
        return f"{canonical}:{Path(path).stem}"

    return canonical


def _require_checkpoint(path: str | None, default_lspi_path: str, agent_name: str) -> str:
    ckpt = path or default_lspi_path

    if not ckpt:
        raise ValueError(
            f"{agent_name} requires --lspi-v1-path, --white-path, or --black-path"
        )

    if not Path(ckpt).exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt}")

    return ckpt


def make_agent(
    kind: str,
    *,
    path: str | None,
    default_lspi_path: str,
    seed: int,
    search_depth: int,
    search_max_branch: int | None,
    search_use_draw_safety: bool,
    search_use_tactical_safety: bool,
) -> Agent:
    canonical = _canonical_kind(kind)

    if canonical == "random":
        return RandomAgent(seed=seed)

    if canonical == "material":
        return MaterialGreedyAgent(seed=seed)

    if canonical == "lspi_plain":
        ckpt = _require_checkpoint(path, default_lspi_path, canonical)
        return LSPIV1Agent.load(ckpt)

    if canonical == "lspi_tactical":
        ckpt = _require_checkpoint(path, default_lspi_path, canonical)
        return LSPITacticalAgent.load(ckpt)

    if canonical == "lspi_search":
        ckpt = _require_checkpoint(path, default_lspi_path, canonical)

        return LSPISearchAgent.load(
            ckpt,
            depth=search_depth,
            max_branch=search_max_branch,
            use_draw_safety=search_use_draw_safety,
            use_tactical_safety=search_use_tactical_safety,
        )

    raise ValueError(f"unknown agent kind: {kind!r}")


def make_random_opening_fen(
    *,
    start_fen: str,
    random_plies: int,
    rng: random.Random,
) -> str:
    board = Board()
    board.init_board(start_fen)

    for _ in range(random_plies):
        done, _reason = board.game_end_state()
        if done:
            break

        moves = board.get_all_legal_moves()
        if not moves:
            break

        board.make_move(rng.choice(moves))

    return board.to_fen()


def _winner_from_terminal(board: Board, reason: str) -> Winner:
    if reason == "checkmate":
        # Side to move is checkmated.
        return "black" if board.get_is_white_to_move() else "white"

    return "draw"


def play_game(
    *,
    game_idx: int,
    start_fen: str,
    white_agent: Agent,
    black_agent: Agent,
    white_label: str,
    black_label: str,
    max_plies: int,
) -> GameResult:
    board = Board()
    board.init_board(start_fen)

    plies = 0

    white_queen_moves_10 = 0
    black_queen_moves_10 = 0

    white_queen_out_10 = False
    black_queen_out_10 = False

    white_castled_20 = False
    black_castled_20 = False

    white_minor_dev_10 = _count_developed_minors(board, True)
    black_minor_dev_10 = _count_developed_minors(board, False)

    white_center_pawns_8 = _count_center_pawns(board, True)
    black_center_pawns_8 = _count_center_pawns(board, False)
    center_snapshot_taken = False

    def result(
        *,
        winner: Winner,
        reason: str,
        illegal_move: str | None = None,
    ) -> GameResult:
        return GameResult(
            game_idx=game_idx,
            white_label=white_label,
            black_label=black_label,
            winner=winner,
            reason=reason,
            plies=plies,
            start_fen=start_fen,
            final_fen=board.to_fen(),
            illegal_move=illegal_move,

            white_queen_moves_10=white_queen_moves_10,
            black_queen_moves_10=black_queen_moves_10,

            white_queen_out_10=white_queen_out_10,
            black_queen_out_10=black_queen_out_10,

            white_castled_20=white_castled_20,
            black_castled_20=black_castled_20,

            white_minor_dev_10=white_minor_dev_10,
            black_minor_dev_10=black_minor_dev_10,

            white_center_pawns_8=white_center_pawns_8,
            black_center_pawns_8=black_center_pawns_8,
        )

    for _ in range(max_plies):
        done, reason = board.game_end_state()
        if done:
            return result(
                winner=_winner_from_terminal(board, reason),
                reason=reason,
            )

        side_white = board.get_is_white_to_move()
        agent = white_agent if side_white else black_agent

        try:
            move = agent.pick_move(board)
        except Exception as e:
            winner: Winner = "black" if side_white else "white"
            return result(
                winner=winner,
                reason=f"agent error: {type(e).__name__}: {e}",
            )

        if move is None:
            winner = "black" if side_white else "white"
            return result(
                winner=winner,
                reason="agent returned no move",
            )

        # ------------------------------------------------------------
        # Opening/style diagnostics: before applying the chosen move
        # ------------------------------------------------------------
        ply_1based = plies + 1
        mover_white = side_white

        moving_piece = _piece_at(board, move.src_square)
        moving_type = p.piece_type(moving_piece)

        if ply_1based <= 10 and moving_type == p.QUEEN:
            if mover_white:
                white_queen_moves_10 += 1
                white_queen_out_10 = True
            else:
                black_queen_moves_10 += 1
                black_queen_out_10 = True

        if ply_1based <= 20 and (move.flags & F_CASTLE):
            if mover_white:
                white_castled_20 = True
            else:
                black_castled_20 = True

        move_text = _move_to_uci_like(board, move)

        ok = board.make_move(move)
        if not ok:
            winner = "black" if side_white else "white"
            return result(
                winner=winner,
                reason="illegal move",
                illegal_move=move_text,
            )

        plies += 1

        # ------------------------------------------------------------
        # Opening/style diagnostics: after applying the move
        # ------------------------------------------------------------
        if plies <= 10:
            white_minor_dev_10 = max(
                white_minor_dev_10,
                _count_developed_minors(board, True),
            )
            black_minor_dev_10 = max(
                black_minor_dev_10,
                _count_developed_minors(board, False),
            )

        if plies >= 8 and not center_snapshot_taken:
            white_center_pawns_8 = _count_center_pawns(board, True)
            black_center_pawns_8 = _count_center_pawns(board, False)
            center_snapshot_taken = True

    return result(
        winner="draw",
        reason="max plies",
    )

def material_cp_from_fen(fen: str) -> float:
    """
    Return white material advantage in centipawn-like units.

    material_potential() uses:
      pawn = 0.1
      knight = 0.32
      bishop = 0.33
      rook = 0.5
      queen = 0.9

    Multiplying by 1000 gives:
      pawn = 100
      knight = 320
      bishop = 330
      rook = 500
      queen = 900
    """
    b = Board()
    b.init_board(fen)
    return float(material_potential(b) * 1000.0)


def material_cp_for_agent(result, agent_name: str) -> float | None:
    """
    Final material from the given agent's perspective.

    Positive means the agent is ahead.
    Negative means the agent is behind.
    """
    white_label = result.white_label
    black_label = result.black_label

    white_cp = material_cp_from_fen(result.final_fen)

    if agent_name == white_label:
        return white_cp

    if agent_name == black_label:
        return -white_cp

    return None


def bucket_material(cp: float, *, equal_threshold_cp: float = 50.0) -> str:
    """
    Treat positions within half a pawn as roughly equal.

    Since pawn = 100 cp, threshold 50 cp means:
      > +50: ahead
      < -50: behind
      otherwise: equal
    """
    if cp > equal_threshold_cp:
        return "ahead"
    if cp < -equal_threshold_cp:
        return "behind"
    return "equal"


def print_draw_material_summary(results) -> None:
    """
    Summarize final material in drawn games from each agent's perspective.
    """
    agents = sorted({r.white_label for r in results} | {r.black_label for r in results})
    drawn = [r for r in results if r.winner == "draw"]

    if not drawn:
        print()
        print("Draw material summary: no drawn games.")
        return

    stats = {
        agent: {
            "draws": 0,
            "ahead": 0,
            "equal": 0,
            "behind": 0,
            "cp_values": [],
            "by_reason": defaultdict(list),
        }
        for agent in agents
    }

    for r in drawn:
        for agent in agents:
            cp = material_cp_for_agent(r, agent)
            if cp is None:
                continue

            bucket = bucket_material(cp)

            s = stats[agent]
            s["draws"] += 1
            s[bucket] += 1
            s["cp_values"].append(cp)
            s["by_reason"][r.reason].append(cp)

    print()
    print("Draw material summary:")
    print("  Positive cp means the agent was materially ahead at the draw.")
    print("  Equal bucket uses ±50 cp.")

    for agent in agents:
        s = stats[agent]
        cps = s["cp_values"]

        if not cps:
            continue

        avg_cp = mean(cps)
        min_cp = min(cps)
        max_cp = max(cps)

        print()
        print(f"  {agent}:")
        print(f"    drawn games: {s['draws']}")
        print(f"    ahead/equal/behind: {s['ahead']}/{s['equal']}/{s['behind']}")
        print(f"    avg final material: {avg_cp:+.1f} cp")
        print(f"    range: {min_cp:+.1f} cp to {max_cp:+.1f} cp")

        print("    by draw reason:")
        for reason, vals in sorted(s["by_reason"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
            vals_avg = mean(vals)
            ahead = sum(1 for x in vals if bucket_material(x) == "ahead")
            equal = sum(1 for x in vals if bucket_material(x) == "equal")
            behind = sum(1 for x in vals if bucket_material(x) == "behind")

            print(
                f"      {reason}: "
                f"n={len(vals)}, "
                f"avg={vals_avg:+.1f} cp, "
                f"ahead/equal/behind={ahead}/{equal}/{behind}"
            )

def print_summary(results: list[GameResult]) -> None:
    n = len(results)
    if n == 0:
        print("No games played.")
        return

    white_wins = sum(1 for r in results if r.winner == "white")
    black_wins = sum(1 for r in results if r.winner == "black")
    draws = sum(1 for r in results if r.winner == "draw")
    avg_plies = sum(r.plies for r in results) / n

    reasons = Counter(r.reason for r in results)

    scores: dict[str, AgentScore] = defaultdict(AgentScore)

    for r in results:
        white_score = scores[r.white_label]
        black_score = scores[r.black_label]

        if r.winner == "white":
            white_score.wins += 1
            black_score.losses += 1
        elif r.winner == "black":
            black_score.wins += 1
            white_score.losses += 1
        else:
            white_score.draws += 1
            black_score.draws += 1

    print()
    print("=== Evaluation summary ===")
    print(f"Games:      {n}")
    print(f"White wins: {white_wins} ({white_wins / n:.1%})")
    print(f"Black wins: {black_wins} ({black_wins / n:.1%})")
    print(f"Draws:      {draws} ({draws / n:.1%})")
    print(f"Avg plies:  {avg_plies:.1f}")

    print()
    print("Reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    print()
    print("Agent scores:")
    for label, score in sorted(scores.items()):
        print(
            f"  {label}: "
            f"{score.wins}W-{score.draws}D-{score.losses}L "
            f"score={score.score:.1%} "
            f"games={score.games}"
        )
    print_draw_material_summary(results)

def save_json(path: str, results: list[GameResult]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "results": [asdict(r) for r in results],
    }

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON results to: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--white",
        required=True,
        help=(
            "White agent. Options/aliases: random, material, "
            "lspi_plain/plain/lspi_v1, "
            "lspi_tactical/tactical/safe, "
            "lspi_search/search."
        ),
    )

    ap.add_argument(
        "--black",
        required=True,
        help=(
            "Black agent. Options/aliases: random, material, "
            "lspi_plain/plain/lspi_v1, "
            "lspi_tactical/tactical/safe, "
            "lspi_search/search."
        ),
    )

    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument(
        "--start-fen",
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )

    ap.add_argument(
        "--random-openings",
        type=int,
        default=0,
        help="Number of random legal plies before evaluation starts.",
    )

    ap.add_argument(
        "--swap-colors",
        action="store_true",
        help="For each start position, also play a second game with agents swapped.",
    )

    ap.add_argument(
        "--lspi-v1-path",
        default="data/processed/checkpoints/lspi_v1.9_profiling.npz",
        help="Default checkpoint path for lspi_v1 agents.",
    )
    ap.add_argument("--white-path", default=None, help="Checkpoint path for white agent if needed.")
    ap.add_argument("--black-path", default=None, help="Checkpoint path for black agent if needed.")

    ap.add_argument(
        "--search-depth",
        type=int,
        default=2,
        help="Depth for lspi_search agents.",
    )

    ap.add_argument(
        "--search-max-branch",
        type=int,
        default=None,
        help="Branch cap for lspi_search agents. Default: None/full width.",
    )

    ap.add_argument(
        "--search-no-draw-safety",
        action="store_true",
        help="Disable draw-safety adjustment for lspi_search agents.",
    )

    ap.add_argument(
        "--search-no-tactical-safety",
        action="store_true",
        help="Disable tactical-safety adjustment for lspi_search agents.",
    )

    ap.add_argument("--json-out", default=None)
    ap.add_argument("--progress-every", type=int, default=10)

    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )

    args = ap.parse_args()

    rng = random.Random(args.seed)

    white_label = _agent_label(args.white, args.white_path)
    black_label = _agent_label(args.black, args.black_path)

    if (
        args.white == args.black
        and args.white_path == args.black_path
        and args.random_openings == 0
        and not args.swap_colors
    ):
        print(
            "[warning] Same deterministic agents from the same start position may repeat "
            "the exact same game. Consider --random-openings or --swap-colors."
        )

    t0 = time.time()

    results: list[GameResult] = []
    game_idx = 1

    # Reuse agent instances across games so feature caches/checkpoint objects can persist.
    base_white_agent = make_agent(
        args.white,
        path=args.white_path,
        default_lspi_path=args.lspi_v1_path,
        seed=args.seed + 101,
        search_depth=args.search_depth,
        search_max_branch=args.search_max_branch,
        search_use_draw_safety=not args.search_no_draw_safety,
        search_use_tactical_safety=not args.search_no_tactical_safety,
    )
    base_black_agent = make_agent(
        args.black,
        path=args.black_path,
        default_lspi_path=args.lspi_v1_path,
        seed=args.seed + 202,
        search_depth=args.search_depth,
        search_max_branch=args.search_max_branch,
        search_use_draw_safety=not args.search_no_draw_safety,
        search_use_tactical_safety=not args.search_no_tactical_safety,
    )

    use_tqdm = (not args.no_progress) and tqdm is not None

    if use_tqdm:
        position_iter = tqdm(
            range(args.games),
            total=args.games,
            desc="eval start positions",
            unit="pos",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    else:
        position_iter = range(args.games)

    for i in position_iter:
        start_fen = make_random_opening_fen(
            start_fen=args.start_fen,
            random_plies=args.random_openings,
            rng=rng,
        )

        # Normal color assignment.
        result = play_game(
            game_idx=game_idx,
            start_fen=start_fen,
            white_agent=base_white_agent,
            black_agent=base_black_agent,
            white_label=white_label,
            black_label=black_label,
            max_plies=args.max_plies,
        )
        results.append(result)
        game_idx += 1

        # Optional swapped-color game from the same start position.
        if args.swap_colors:
            result = play_game(
                game_idx=game_idx,
                start_fen=start_fen,
                white_agent=base_black_agent,
                black_agent=base_white_agent,
                white_label=black_label,
                black_label=white_label,
                max_plies=args.max_plies,
            )
            results.append(result)
            game_idx += 1

        if use_tqdm:
            reason_counts = Counter(r.reason for r in results)

            played = len(results)
            draws = sum(1 for r in results if r.winner == "draw")
            checkmates = reason_counts.get("checkmate", 0)
            repetitions = reason_counts.get("threefold repetition", 0)
            stalemates = reason_counts.get("stalemate", 0)
            fifty = reason_counts.get("fifty-move rule", 0)
            max_plies_count = reason_counts.get("max plies", 0)

            position_iter.set_postfix(
                mate=checkmates,
                draw=draws,
                rep=repetitions,
                stale=stalemates,
                fifty=fifty,
                max=max_plies_count,
            )
        else:
            if args.progress_every and (i + 1) % args.progress_every == 0:
                print(f"Completed {i + 1}/{args.games} start positions...")

    dt = time.time() - t0

    print_summary(results)
    print(f"\nWall time: {dt:.2f}s")
    print(f"Games/sec: {len(results) / max(dt, 1e-9):.2f}")

    if args.json_out:
        save_json(args.json_out, results)

CENTER_SQUARES = {
    27,  # d4
    28,  # e4
    35,  # d5
    36,  # e5
}


def _piece_at(board: Board, sq: int) -> int:
    piece_at = getattr(board, "piece_at", None)
    if piece_at is not None:
        return int(piece_at(sq))
    return int(board.get_board()[sq])


def _count_center_pawns(board: Board, white: bool) -> int:
    target = p.WHITE_PAWN if white else p.BLACK_PAWN
    return sum(1 for sq in CENTER_SQUARES if _piece_at(board, sq) == target)


def _count_developed_minors(board: Board, white: bool) -> int:
    """
    Count bishops/knights not on their home rank.

    White home rank: rank 1 => internal rank 0.
    Black home rank: rank 8 => internal rank 7.

    This is a simple style metric, not a perfect chess concept.
    """
    home_rank = 0 if white else 7
    count = 0

    for sq, pc_raw in enumerate(board.get_board()):
        pc = int(pc_raw)
        if pc == p.NONE:
            continue

        if p.is_white(pc) != white:
            continue

        t = p.piece_type(pc)
        if t not in (p.KNIGHT, p.BISHOP):
            continue

        rank = sq >> 3
        if rank != home_rank:
            count += 1

    return count


if __name__ == "__main__":
    main()