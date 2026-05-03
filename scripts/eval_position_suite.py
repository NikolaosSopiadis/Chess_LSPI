# scripts/eval_position_suite.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.material_greedy import MaterialGreedyAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent
from chess_rl.features.registry import get as get_features


@dataclass(frozen=True, slots=True)
class PositionCase:
    name: str
    fen: str
    expected: frozenset[str]
    category: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class CaseResult:
    name: str
    category: str
    passed: bool
    chosen: str
    expected: tuple[str, ...]
    rank: Optional[int]
    legal_moves: int
    note: str


def move_to_uci_like(board: Board, move: Move) -> str:
    s = board.idx_to_algebraic(move.src_square) + board.idx_to_algebraic(move.dst_square)
    if move.promotion:
        # Your current debug format uses numeric promotion. Avoid promotion tests for now.
        s += f"={move.promotion}"
    return s


def make_agent(kind: str, *, path: str | None, seed: int) -> Agent:
    kind = kind.lower().strip()

    if kind == "random":
        return RandomAgent(seed=seed)

    if kind in ("material", "material_greedy", "greedy_material"):
        return MaterialGreedyAgent(seed=seed)

    if kind in ("lspi", "lspi_v1"):
        if not path:
            raise ValueError("lspi_v1 requires --path")
        if not Path(path).exists():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return LSPIV1Agent.load(path)

    raise ValueError(f"unknown agent kind: {kind!r}")


def built_in_cases() -> list[PositionCase]:
    return [
        # ------------------------------------------------------------------
        # Obvious material captures
        # ------------------------------------------------------------------
        PositionCase(
            name="white_rook_takes_free_queen",
            category="material",
            fen="q3k3/8/8/8/8/8/8/R3K3 w - - 0 1",
            expected=frozenset({"a1a8"}),
            note="White rook can capture an undefended black queen.",
        ),
        PositionCase(
            name="black_rook_takes_free_queen",
            category="material",
            fen="r3k3/8/8/8/8/8/8/Q3K3 b - - 0 1",
            expected=frozenset({"a8a1"}),
            note="Black rook can capture an undefended white queen.",
        ),
        PositionCase(
            name="white_pawn_takes_free_queen",
            category="material",
            fen="4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
            expected=frozenset({"e4d5"}),
            note="White pawn can capture a queen.",
        ),
        PositionCase(
            name="black_pawn_takes_free_queen",
            category="material",
            fen="4k3/8/8/4p3/3Q4/8/8/4K3 b - - 0 1",
            expected=frozenset({"e5d4"}),
            note="Black pawn can capture a queen.",
        ),
        PositionCase(
            name="white_queen_takes_free_queen",
            category="material",
            fen="4k3/8/8/7q/8/8/8/3QK3 w - - 0 1",
            expected=frozenset({"d1h5"}),
            note="White queen can capture black queen on a diagonal.",
        ),
        PositionCase(
            name="black_queen_takes_free_queen",
            category="material",
            fen="3qk3/8/8/8/7Q/8/8/4K3 b - - 0 1",
            expected=frozenset({"d8h4"}),
            note="Black queen can capture white queen on a diagonal.",
        ),

        # ------------------------------------------------------------------
        # Mate in one
        # ------------------------------------------------------------------
        PositionCase(
            name="white_rook_mate_in_one",
            category="mate",
            fen="7k/6pp/8/8/8/8/6PP/5RK1 w - - 0 1",
            expected=frozenset({"f1f8"}),
            note="White has Rf8#.",
        ),
        PositionCase(
            name="black_rook_mate_in_one",
            category="mate",
            fen="5rk1/6pp/8/8/8/8/6PP/7K b - - 0 1",
            expected=frozenset({"f8f1"}),
            note="Black has Rf1#.",
        ),

        # ------------------------------------------------------------------
        # Simple checking/capturing moves
        # ------------------------------------------------------------------
        PositionCase(
            name="white_rook_gives_file_check",
            category="check",
            fen="4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
            expected=frozenset({"a1a8"}),
            note="White rook gives check along the a-file.",
        ),
        PositionCase(
            name="black_rook_gives_file_check",
            category="check",
            fen="r3k3/8/8/8/8/8/8/4K3 b - - 0 1",
            expected=frozenset({"a8a1"}),
            note="Black rook gives check along the a-file.",
        ),
        PositionCase(
            name="white_rook_gives_file_check_nonmate",
            category="check",
            fen="4k3/8/8/8/8/8/5PPP/R3K3 w - - 0 1",
            expected=frozenset({"a1a8"}),
            note="White rook gives check, but Black has escape squares/pieces.",
        ),
        PositionCase(
            name="black_rook_gives_file_check_nonmate",
            category="check",
            fen="r3k3/5ppp/8/8/8/8/8/4K3 b - - 0 1",
            expected=frozenset({"a8a1"}),
            note="Black rook gives check, but White has escape squares/pieces.",
        ),
    ]


def rank_moves_lspi(agent: Agent, board: Board) -> list[tuple[str, float]]:
    """
    Return moves ranked from chosen side's perspective.

    This assumes LSPIV1Agent has:
      - .w
      - .feature_name

    White maximizes w @ phi.
    Black minimizes w @ phi.
    """
    if not isinstance(agent, LSPIV1Agent):
        return []

    feats = get_features(agent.feature_name)
    w = np.asarray(agent.w, dtype=np.float64)

    white_to_move = board.get_is_white_to_move()

    rows: list[tuple[str, float]] = []
    for m in board.get_all_legal_moves():
        phi = feats.phi_sa(board, m)
        score = float(w @ phi)
        rows.append((move_to_uci_like(board, m), score))

    rows.sort(key=lambda x: x[1], reverse=white_to_move)
    return rows


def evaluate_case(
    case: PositionCase,
    agent: Agent,
    *,
    show_top: int = 0,
) -> CaseResult:
    board = Board()
    board.init_board(case.fen)

    legal_moves = board.get_all_legal_moves()
    if not legal_moves:
        return CaseResult(
            name=case.name,
            category=case.category,
            passed=False,
            chosen="<no legal moves>",
            expected=tuple(sorted(case.expected)),
            rank=None,
            legal_moves=0,
            note=case.note,
        )

    move = agent.pick_move(board)
    chosen = move_to_uci_like(board, move)
    passed = chosen in case.expected

    rank: Optional[int] = None
    ranked = rank_moves_lspi(agent, board)

    if ranked:
        for i, (move_text, _score) in enumerate(ranked, start=1):
            if move_text in case.expected:
                rank = i
                break

    if show_top > 0 and ranked:
        print()
        print(f"--- {case.name} ---")
        print(f"FEN: {case.fen}")
        print(f"Expected: {', '.join(sorted(case.expected))}")
        print(f"Chosen:   {chosen}")
        print(f"Pass:     {passed}")
        print(f"Best expected rank: {rank}")
        print("Top moves:")
        for i, (move_text, score) in enumerate(ranked[:show_top], start=1):
            mark = "*" if move_text in case.expected else " "
            print(f"  {i:2d}. {mark} {move_text:8s} score={score:+.6f}")

    return CaseResult(
        name=case.name,
        category=case.category,
        passed=passed,
        chosen=chosen,
        expected=tuple(sorted(case.expected)),
        rank=rank,
        legal_moves=len(legal_moves),
        note=case.note,
    )


def print_summary(results: list[CaseResult]) -> None:
    total = len(results)
    passed = sum(r.passed for r in results)

    print()
    print("=== Position suite summary ===")
    print(f"Cases:  {total}")
    print(f"Passed: {passed}/{total} ({passed / max(1, total):.1%})")

    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    print()
    print("By category:")
    for category, rows in sorted(by_cat.items()):
        p = sum(r.passed for r in rows)
        print(f"  {category:10s}: {p}/{len(rows)} ({p / max(1, len(rows)):.1%})")

    print()
    print("Cases:")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        rank_text = "-" if r.rank is None else str(r.rank)
        exp = ",".join(r.expected)
        print(
            f"  [{status}] {r.name:32s} "
            f"chosen={r.chosen:8s} expected={exp:18s} "
            f"rank={rank_text:>3s} legal={r.legal_moves}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["random", "material", "lspi_v1"])
    ap.add_argument("--path", default=None, help="Checkpoint path for lspi_v1.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-top", type=int, default=0, help="For lspi_v1, print top K ranked moves per case.")
    args = ap.parse_args()

    agent = make_agent(args.agent, path=args.path, seed=args.seed)

    cases = built_in_cases()
    validate_suite(cases)

    results: list[CaseResult] = []
    for case in cases:
        result = evaluate_case(case, agent, show_top=args.show_top)
        results.append(result)

    print_summary(results)
    
    
def validate_suite(cases: list[PositionCase]) -> None:
    bad = False

    for case in cases:
        board = Board()
        board.init_board(case.fen)

        legal = {move_to_uci_like(board, m) for m in board.get_all_legal_moves()}
        missing = case.expected - legal

        if missing:
            bad = True
            print()
            print(f"[invalid test case] {case.name}")
            print(f"FEN: {case.fen}")
            print(f"Expected but not legal: {sorted(missing)}")
            print(f"Legal moves: {sorted(legal)}")

    if bad:
        raise SystemExit("Position suite contains invalid expected moves.")


if __name__ == "__main__":
    main()