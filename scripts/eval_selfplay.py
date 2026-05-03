from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Literal

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.material_greedy import MaterialGreedyAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent


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


def _agent_label(kind: str, path: str | None) -> str:
    if path:
        return f"{kind}:{Path(path).stem}"
    return kind


def make_agent(kind: str, *, path: str | None, default_lspi_path: str, seed: int) -> Agent:
    kind = kind.lower().strip()

    if kind == "random":
        return RandomAgent(seed=seed)

    if kind in ("material", "material_greedy", "greedy_material"):
        return MaterialGreedyAgent(seed=seed)

    if kind in ("lspi", "lspi_v1"):
        ckpt = path or default_lspi_path
        if not ckpt:
            raise ValueError("lspi_v1 requires --lspi-v1-path, --white-path, or --black-path")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint does not exist: {ckpt}")
        return LSPIV1Agent.load(ckpt)

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

    for _ in range(max_plies):
        done, reason = board.game_end_state()
        if done:
            return GameResult(
                game_idx=game_idx,
                white_label=white_label,
                black_label=black_label,
                winner=_winner_from_terminal(board, reason),
                reason=reason,
                plies=plies,
                start_fen=start_fen,
                final_fen=board.to_fen(),
            )

        side_white = board.get_is_white_to_move()
        agent = white_agent if side_white else black_agent

        try:
            move = agent.pick_move(board)
        except Exception as e:
            # Agent failed on its turn, so it loses.
            winner: Winner = "black" if side_white else "white"
            return GameResult(
                game_idx=game_idx,
                white_label=white_label,
                black_label=black_label,
                winner=winner,
                reason=f"agent error: {type(e).__name__}: {e}",
                plies=plies,
                start_fen=start_fen,
                final_fen=board.to_fen(),
            )

        move_text = _move_to_uci_like(board, move)

        ok = board.make_move(move)
        if not ok:
            # Illegal move loses.
            winner = "black" if side_white else "white"
            return GameResult(
                game_idx=game_idx,
                white_label=white_label,
                black_label=black_label,
                winner=winner,
                reason="illegal move",
                plies=plies,
                start_fen=start_fen,
                final_fen=board.to_fen(),
                illegal_move=move_text,
            )

        plies += 1

    return GameResult(
        game_idx=game_idx,
        white_label=white_label,
        black_label=black_label,
        winner="draw",
        reason="max plies",
        plies=plies,
        start_fen=start_fen,
        final_fen=board.to_fen(),
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

    ap.add_argument("--white", required=True, choices=["random", "material", "lspi_v1"])
    ap.add_argument("--black", required=True, choices=["random", "material", "lspi_v1"])

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

    ap.add_argument("--json-out", default=None)
    ap.add_argument("--progress-every", type=int, default=10)

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
    )
    base_black_agent = make_agent(
        args.black,
        path=args.black_path,
        default_lspi_path=args.lspi_v1_path,
        seed=args.seed + 202,
    )

    for i in range(args.games):
        if args.random_openings > 0:
            start_fen = make_random_opening_fen(
                start_fen=args.start_fen,
                random_plies=args.random_openings,
                rng=rng,
            )
        else:
            start_fen = args.start_fen

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

        if args.swap_colors:
            swapped_result = play_game(
                game_idx=game_idx,
                start_fen=start_fen,
                white_agent=base_black_agent,
                black_agent=base_white_agent,
                white_label=black_label,
                black_label=white_label,
                max_plies=args.max_plies,
            )
            results.append(swapped_result)
            game_idx += 1

        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"Completed {i + 1}/{args.games} start positions...")

    dt = time.time() - t0

    print_summary(results)
    print(f"\nWall time: {dt:.2f}s")
    print(f"Games/sec: {len(results) / max(dt, 1e-9):.2f}")

    if args.json_out:
        save_json(args.json_out, results)


if __name__ == "__main__":
    main()