from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.material_greedy import MaterialGreedyAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent
from chess_rl.agents.lspi_tactical import LSPITacticalAgent
from chess_rl.features.registry import get as get_features


try:
    from chess_rl.agents.lspi_search import LSPISearchAgent
except Exception:
    LSPISearchAgent = None


@dataclass(frozen=True)
class SuiteCase:
    name: str
    fen: str
    good: list[str]
    bad: list[str]
    tags: list[str]
    note: str = ""


@dataclass
class CaseResult:
    name: str
    tags: list[str]
    actual_move: str | None
    actual_is_good: bool
    best_good_rank: int | None
    best_bad_rank: int | None
    bad_above_good: bool
    best_good_score: float | None
    actual_raw_score: float | None
    score_loss_to_best_good: float | None


def built_in_cases() -> list[SuiteCase]:
    """
    Small, deliberately simple king-safety/development suite.

    These are not meant to be a complete chess test.
    They are diagnostic probes:
      - does the model rank castling sensibly?
      - does it avoid silly king walks?
      - does it prefer minor development over early queen wandering?
    """
    return [
        SuiteCase(
            name="white_castle_basic",
            fen="rnbqkbnr/pppppppp/8/8/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
            good=["e1g1"],
            bad=["e1e2"],
            tags=["castle", "white", "opening", "avoid_king_walk"],
            note="White can castle kingside. Ke2 is the kind of move we want to avoid.",
        ),
        SuiteCase(
            name="black_castle_basic",
            fen="rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
            good=["e8g8"],
            bad=["e8e7"],
            tags=["castle", "black", "opening", "avoid_king_walk"],
            note="Black can castle kingside. Ke7 is the kind of move we want to avoid.",
        ),
        SuiteCase(
            name="white_castle_either_side",
            fen="r3kbnr/pppqpppp/2np4/8/8/2NPBN2/PPPQPPPP/R3K2R w KQkq - 0 6",
            good=["e1g1", "e1c1"],
            bad=["e1e2"],
            tags=["castle", "white", "opening", "avoid_king_walk"],
            note="White has both castling options available.",
        ),
        SuiteCase(
            name="white_develop_before_queen",
            fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            good=["f1c4", "f1b5", "b1c3", "d2d4"],
            bad=["d1e2", "d1f3"],
            tags=["development", "white", "opening", "queen"],
            note="White should usually develop/castle-plan rather than shuffle queen early.",
        ),
        SuiteCase(
            name="black_develop_before_queen",
            fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3",
            good=["g8f6", "f8c5", "f8b4", "d7d6"],
            bad=["d8h4", "d8f6"],
            tags=["development", "black", "opening", "queen"],
            note="Black should usually develop rather than launch the queen early.",
        ),
        SuiteCase(
            name="white_castle_after_development",
            fen="r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 5",
            good=["e1g1"],
            bad=["e1e2", "d1e2"],
            tags=["castle", "white", "development", "opening"],
            note="A normal Italian-like position where castling is natural.",
        ),
        SuiteCase(
            name="black_castle_after_development",
            fen="r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 4 5",
            good=["e8g8"],
            bad=["e8e7", "d8e7"],
            tags=["castle", "black", "development", "opening"],
            note="A normal position where Black should be happy to castle.",
        ),
    ]


def load_cases(path: str | None) -> list[SuiteCase]:
    if path is None:
        return built_in_cases()

    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    cases: list[SuiteCase] = []

    for item in payload:
        cases.append(
            SuiteCase(
                name=item["name"],
                fen=item["fen"],
                good=list(item.get("good", [])),
                bad=list(item.get("bad", [])),
                tags=list(item.get("tags", [])),
                note=item.get("note", ""),
            )
        )

    return cases


def move_to_uci(board: Board, move: Move) -> str:
    text = board.idx_to_algebraic(move.src_square) + board.idx_to_algebraic(move.dst_square)

    # Promotion diagnostics are not central in this suite.
    # Keep a suffix if your Move object stores one.
    if getattr(move, "promotion", 0):
        text += f"={move.promotion}"

    return text


def move_matches_uci(board: Board, move: Move, uci: str) -> bool:
    # Accept prefix match so e7e8 also matches e7e8=Q-like variants.
    return move_to_uci(board, move).startswith(uci)


def find_legal_move(board: Board, uci: str) -> Move | None:
    for move in board.get_all_legal_moves():
        if move_matches_uci(board, move, uci):
            return move

    return None


def canonical_kind(kind: str) -> str:
    k = kind.lower().strip()

    if k == "random":
        return "random"

    if k in ("material", "material_greedy", "greedy_material"):
        return "material"

    if k in ("plain", "lspi_plain", "lspi_v1"):
        return "plain"

    if k in ("tactical", "safe", "lspi_tactical", "lspi_safe"):
        return "tactical"

    if k in ("search", "lspi_search"):
        return "search"

    raise ValueError(f"unknown agent kind: {kind!r}")


def make_agent(
    *,
    kind: str,
    path: str | None,
    seed: int,
    search_depth: int,
    search_max_branch: int | None,
    search_use_draw_safety: bool,
    search_use_tactical_safety: bool,
) -> Agent:
    k = canonical_kind(kind)

    if k == "random":
        return RandomAgent(seed=seed)

    if k == "material":
        return MaterialGreedyAgent(seed=seed)

    if path is None:
        raise ValueError(f"{k} agent requires --path")

    if not Path(path).exists():
        raise FileNotFoundError(path)

    if k == "plain":
        return LSPIV1Agent.load(path)

    if k == "tactical":
        return LSPITacticalAgent.load(path)

    if k == "search":
        if LSPISearchAgent is None:
            raise RuntimeError("Could not import chess_rl.agents.lspi_search.LSPISearchAgent")

        return LSPISearchAgent.load(
            path,
            depth=search_depth,
            max_branch=search_max_branch,
            use_draw_safety=search_use_draw_safety,
            use_tactical_safety=search_use_tactical_safety,
        )

    raise AssertionError(k)


def load_checkpoint_info(path: str) -> tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=False)
    w = data["w"]

    meta_raw = str(data["meta"])
    meta = json.loads(meta_raw)

    feature_name = meta.get("feature_name", "v1_basic")
    return w, feature_name


def raw_rank_moves(
    *,
    board: Board,
    w: np.ndarray,
    feature_name: str,
) -> list[tuple[str, float, Move]]:
    feats = get_features(feature_name)
    white_to_move = board.get_is_white_to_move()

    scored: list[tuple[str, float, Move]] = []

    for move in board.get_all_legal_moves():
        with board.temporary_move(move):
            phi = feats.phi_afterstate(board)
            score = float(w @ phi)

        scored.append((move_to_uci(board, move), score, move))

    # White maximizes, Black minimizes.
    scored.sort(key=lambda x: x[1], reverse=white_to_move)
    return scored


def rank_of_move(ranked: list[tuple[str, float, Move]], uci: str) -> int | None:
    for i, (move_text, _score, _move) in enumerate(ranked, start=1):
        if move_text.startswith(uci):
            return i

    return None


def score_of_move(ranked: list[tuple[str, float, Move]], uci: str) -> float | None:
    for move_text, score, _move in ranked:
        if move_text.startswith(uci):
            return score

    return None


def best_good_rank(ranked: list[tuple[str, float, Move]], good: list[str]) -> int | None:
    ranks = [rank_of_move(ranked, uci) for uci in good]
    ranks = [r for r in ranks if r is not None]

    if not ranks:
        return None

    return min(ranks)


def best_bad_rank(ranked: list[tuple[str, float, Move]], bad: list[str]) -> int | None:
    ranks = [rank_of_move(ranked, uci) for uci in bad]
    ranks = [r for r in ranks if r is not None]

    if not ranks:
        return None

    return min(ranks)


def best_good_score(
    *,
    ranked: list[tuple[str, float, Move]],
    good: list[str],
    white_to_move: bool,
) -> float | None:
    scores = [score_of_move(ranked, uci) for uci in good]
    scores = [s for s in scores if s is not None]

    if not scores:
        return None

    if white_to_move:
        return max(scores)

    return min(scores)


def score_loss_to_good(
    *,
    chosen_score: float | None,
    good_score: float | None,
    white_to_move: bool,
) -> float | None:
    if chosen_score is None or good_score is None:
        return None

    if white_to_move:
        return good_score - chosen_score

    return chosen_score - good_score


def evaluate_case(
    *,
    case: SuiteCase,
    agent: Agent,
    w: np.ndarray | None,
    feature_name: str | None,
    show_top: int,
) -> CaseResult:
    board = Board()
    board.init_board(case.fen)

    legal_texts = [move_to_uci(board, move) for move in board.get_all_legal_moves()]

    missing_good = [uci for uci in case.good if find_legal_move(board, uci) is None]
    missing_bad = [uci for uci in case.bad if find_legal_move(board, uci) is None]

    if missing_good:
        print(f"[warning] {case.name}: expected good moves not legal/found: {missing_good}")
    if missing_bad:
        print(f"[warning] {case.name}: expected bad moves not legal/found: {missing_bad}")

    actual_move_text: str | None = None
    actual_is_good = False

    try:
        actual_move = agent.pick_move(board)
        actual_move_text = move_to_uci(board, actual_move)
        actual_is_good = any(actual_move_text.startswith(uci) for uci in case.good)
    except Exception as e:
        actual_move_text = f"ERROR:{type(e).__name__}:{e}"
        actual_is_good = False

    ranked: list[tuple[str, float, Move]] = []

    best_g_rank: int | None = None
    best_b_rank: int | None = None
    bad_above = False
    best_g_score: float | None = None
    actual_raw_score: float | None = None
    loss: float | None = None

    if w is not None and feature_name is not None:
        ranked = raw_rank_moves(board=board, w=w, feature_name=feature_name)
        best_g_rank = best_good_rank(ranked, case.good)
        best_b_rank = best_bad_rank(ranked, case.bad)

        if best_g_rank is not None and best_b_rank is not None:
            bad_above = best_b_rank < best_g_rank

        best_g_score = best_good_score(
            ranked=ranked,
            good=case.good,
            white_to_move=board.get_is_white_to_move(),
        )

        if actual_move_text is not None and not actual_move_text.startswith("ERROR:"):
            actual_raw_score = score_of_move(ranked, actual_move_text[:4])

        loss = score_loss_to_good(
            chosen_score=actual_raw_score,
            good_score=best_g_score,
            white_to_move=board.get_is_white_to_move(),
        )

    print()
    print("=" * 100)
    print(case.name)
    print("=" * 100)
    print(f"tags:     {', '.join(case.tags)}")
    print(f"note:     {case.note}")
    print(f"fen:      {case.fen}")
    print(f"good:     {case.good}")
    print(f"bad:      {case.bad}")
    print(f"actual:   {actual_move_text} {'✓' if actual_is_good else '✗'}")

    if ranked:
        print(f"raw best good rank: {best_g_rank}")
        print(f"raw best bad rank:  {best_b_rank}")
        print(f"bad above good:     {bad_above}")
        if loss is not None:
            print(f"raw score loss to best good: {loss:+.6f}")

        print()
        print(f"top {show_top} raw checkpoint moves:")
        for i, (move_text, score, _move) in enumerate(ranked[:show_top], start=1):
            flags: list[str] = []
            if any(move_text.startswith(uci) for uci in case.good):
                flags.append("GOOD")
            if any(move_text.startswith(uci) for uci in case.bad):
                flags.append("BAD")
            if actual_move_text and move_text.startswith(actual_move_text[:4]):
                flags.append("ACTUAL")

            suffix = f"  [{' '.join(flags)}]" if flags else ""
            print(f"  {i:2d}. {move_text:8s} {score:+.6f}{suffix}")

    else:
        print("raw checkpoint ranking: unavailable")

    if not ranked:
        print()
        print(f"legal moves: {legal_texts}")

    return CaseResult(
        name=case.name,
        tags=case.tags,
        actual_move=actual_move_text,
        actual_is_good=actual_is_good,
        best_good_rank=best_g_rank,
        best_bad_rank=best_b_rank,
        bad_above_good=bad_above,
        best_good_score=best_g_score,
        actual_raw_score=actual_raw_score,
        score_loss_to_best_good=loss,
    )


def print_summary(results: list[CaseResult]) -> None:
    if not results:
        print("No results.")
        return

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    actual_good = sum(1 for r in results if r.actual_is_good)
    top1 = sum(1 for r in results if r.best_good_rank == 1)
    top3 = sum(1 for r in results if r.best_good_rank is not None and r.best_good_rank <= 3)
    top5 = sum(1 for r in results if r.best_good_rank is not None and r.best_good_rank <= 5)
    bad_above = sum(1 for r in results if r.bad_above_good)

    ranks = [r.best_good_rank for r in results if r.best_good_rank is not None]
    losses = [
        r.score_loss_to_best_good
        for r in results
        if r.score_loss_to_best_good is not None
    ]

    print(f"cases:                 {len(results)}")
    print(f"actual chose good:     {actual_good}/{len(results)} ({actual_good / len(results):.1%})")
    print(f"raw good top-1:        {top1}/{len(results)} ({top1 / len(results):.1%})")
    print(f"raw good top-3:        {top3}/{len(results)} ({top3 / len(results):.1%})")
    print(f"raw good top-5:        {top5}/{len(results)} ({top5 / len(results):.1%})")
    print(f"bad ranked over good:  {bad_above}/{len(results)} ({bad_above / len(results):.1%})")

    if ranks:
        print(f"avg best-good rank:    {mean(ranks):.2f}")

    if losses:
        print(f"avg score loss:        {mean(losses):+.6f}")

    by_tag: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        for tag in r.tags:
            by_tag[tag].append(r)

    print()
    print("By tag:")

    for tag, rows in sorted(by_tag.items()):
        n = len(rows)
        tag_actual_good = sum(1 for r in rows if r.actual_is_good)
        tag_top3 = sum(
            1
            for r in rows
            if r.best_good_rank is not None and r.best_good_rank <= 3
        )
        tag_bad_above = sum(1 for r in rows if r.bad_above_good)

        tag_ranks = [r.best_good_rank for r in rows if r.best_good_rank is not None]

        avg_rank = mean(tag_ranks) if tag_ranks else float("nan")

        print(
            f"  {tag:18s} "
            f"n={n:2d} "
            f"actual_good={tag_actual_good}/{n} "
            f"top3={tag_top3}/{n} "
            f"bad>good={tag_bad_above}/{n} "
            f"avg_rank={avg_rank:.2f}"
        )


def save_json(path: str, results: list[CaseResult]) -> None:
    payload = {
        "results": [
            {
                "name": r.name,
                "tags": r.tags,
                "actual_move": r.actual_move,
                "actual_is_good": r.actual_is_good,
                "best_good_rank": r.best_good_rank,
                "best_bad_rank": r.best_bad_rank,
                "bad_above_good": r.bad_above_good,
                "best_good_score": r.best_good_score,
                "actual_raw_score": r.actual_raw_score,
                "score_loss_to_best_good": r.score_loss_to_best_good,
            }
            for r in results
        ]
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--agent",
        default="tactical",
        help="Agent kind: plain/lspi_v1, tactical/safe, search, material, random.",
    )

    ap.add_argument(
        "--path",
        default=None,
        help="Checkpoint path for LSPI agents.",
    )

    ap.add_argument(
        "--cases-json",
        default=None,
        help="Optional JSON file with custom suite cases.",
    )

    ap.add_argument("--show-top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json-out", default=None)

    ap.add_argument("--search-depth", type=int, default=2)
    ap.add_argument("--search-max-branch", type=int, default=None)
    ap.add_argument("--search-no-draw-safety", action="store_true")
    ap.add_argument("--search-no-tactical-safety", action="store_true")

    args = ap.parse_args()

    agent = make_agent(
        kind=args.agent,
        path=args.path,
        seed=args.seed,
        search_depth=args.search_depth,
        search_max_branch=args.search_max_branch,
        search_use_draw_safety=not args.search_no_draw_safety,
        search_use_tactical_safety=not args.search_no_tactical_safety,
    )

    w: np.ndarray | None = None
    feature_name: str | None = None

    if args.path is not None:
        w, feature_name = load_checkpoint_info(args.path)
        print(f"checkpoint: {args.path}")
        print(f"feature:    {feature_name}")
        print(f"dim:        {len(w)}")
    else:
        print("checkpoint: none")
        print("raw ranking disabled")

    cases = load_cases(args.cases_json)

    results: list[CaseResult] = []

    for case in cases:
        result = evaluate_case(
            case=case,
            agent=agent,
            w=w,
            feature_name=feature_name,
            show_top=args.show_top,
        )
        results.append(result)

    print_summary(results)

    if args.json_out:
        save_json(args.json_out, results)


if __name__ == "__main__":
    main()