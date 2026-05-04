from __future__ import annotations

import argparse
import gzip
import io
import json
import time
from pathlib import Path
from typing import TextIO, Any

import chess
import chess.pgn
from tqdm import tqdm

try:
    import zstandard as zstd  # type: ignore
except Exception:
    zstd = None  # type: ignore

from chess_core.board import Board
from chess_core.move import (
    Move,
    PROMO_NONE,
    PROMO_KNIGHT,
    PROMO_BISHOP,
    PROMO_ROOK,
    PROMO_QUEEN,
)
from chess_rl.features.registry import get as get_features
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


REWARD_VERSION = "pgn_terminal_plus_potential_v1"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_UCI_PROMO = {
    "n": PROMO_KNIGHT,
    "b": PROMO_BISHOP,
    "r": PROMO_ROOK,
    "q": PROMO_QUEEN,
}


def open_pgn_text(path: str) -> TextIO:
    """
    Open .pgn, .pgn.gz, or .pgn.zst as text.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("Reading .zst requires: pip install zstandard")

        raw_f = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(raw_f)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")

    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_int_header(headers: chess.pgn.Headers, key: str) -> int | None:
    value = headers.get(key)
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def result_to_white_reward(result: str, scale: float) -> float | None:
    """
    Convert PGN result to white-perspective terminal reward.
    """
    if result == "1-0":
        return +scale
    if result == "0-1":
        return -scale
    if result == "1/2-1/2":
        return 0.0
    return None


def game_start_fen(game: chess.pgn.Game) -> str:
    """
    Support FEN-starting games, but default to normal chess.
    """
    fen = game.headers.get("FEN")
    if fen:
        return fen
    return START_FEN


def is_supported_game(game: chess.pgn.Game) -> bool:
    """
    Skip unsupported variants.

    This custom engine currently assumes normal 8x8 standard chess.
    Lichess standard PGN dumps should usually be fine, but this guard
    prevents accidental Chess960/variant ingestion.
    """
    variant = game.headers.get("Variant", "")
    if variant and variant.lower() not in ("standard", "chess"):
        return False

    return True


def uci_to_move_pseudo(board: Board, uci: str) -> Move:
    """
    Fast PGN ingestion path.

    Unlike chess_rl.uci.uci_to_move(), this matches against pseudolegal
    moves instead of legal moves. This is okay for PGN ingestion because
    python-chess has already parsed the game as legal.

    If this fails, your custom Board state has diverged from python-chess.
    """
    uci = uci.strip()

    if len(uci) not in (4, 5):
        raise ValueError(f"Bad UCI: {uci!r}")

    src_sq = board.algebraic_to_idx(uci[0:2])
    dst_sq = board.algebraic_to_idx(uci[2:4])

    promo = PROMO_NONE
    if len(uci) == 5:
        ch = uci[4].lower()
        if ch not in _UCI_PROMO:
            raise ValueError(f"Bad UCI promotion: {uci!r}")
        promo = _UCI_PROMO[ch]

    for m in board.get_pseudolegal_moves(src_sq):
        if m.dst_square != dst_sq:
            continue

        if promo != PROMO_NONE and m.promotion != promo:
            continue

        if promo == PROMO_NONE and m.promotion != PROMO_NONE:
            continue

        return m

    raise ValueError(
        f"Could not match PGN move {uci!r} in custom board position {board.to_fen()!r}"
    )


def process_game(
    game: chess.pgn.Game,
    *,
    feats: Any,
    reward_alpha: float,
    result_scale: float,
    min_elo: int | None,
    max_plies: int | None,
    keep_metadata: bool,
) -> list[dict]:
    """
    Convert one PGN game into LSPI samples.

    Fast path:
    - python-chess parses/validates PGN legality.
    - custom engine matches each move pseudolegally.
    - no game_end_state() per ply.
    - no phi_sa() do/undo; instead apply once and use phi_afterstate().
    - terminal reward comes from the PGN result on the final emitted move.
    """
    if not is_supported_game(game):
        return []

    result = game.headers.get("Result", "*")
    terminal_from_result = result_to_white_reward(result, result_scale)

    if terminal_from_result is None:
        return []

    white_elo = parse_int_header(game.headers, "WhiteElo")
    black_elo = parse_int_header(game.headers, "BlackElo")

    if min_elo is not None:
        if white_elo is None or black_elo is None:
            return []
        if min(white_elo, black_elo) < min_elo:
            return []

    start_fen = game_start_fen(game)

    my_board = Board()
    try:
        my_board.init_board(start_fen)
    except Exception:
        return []

    all_moves = list(game.mainline_moves())
    if not all_moves:
        return []

    truncated = False
    if max_plies is not None and len(all_moves) > max_plies:
        all_moves = all_moves[:max_plies]
        truncated = True

    out: list[dict] = []

    for ply_index, py_move in enumerate(all_moves, start=1):
        fen_before = my_board.to_fen() if keep_metadata else ""
        white_to_move_before = my_board.get_is_white_to_move()
        pot0 = material_potential(my_board)

        uci = py_move.uci()

        try:
            my_move = uci_to_move_pseudo(my_board, uci)
        except Exception:
            # Custom engine diverged from python-chess. Skip whole game.
            return []

        # Apply once.
        my_board._do_move(my_move)

        pot1 = material_potential(my_board)
        phi = feats.phi_afterstate(my_board)
        fen_next = my_board.to_fen()

        is_true_final_game_move = (not truncated) and (ply_index == len(all_moves))

        r_material = reward_alpha * (pot1 - pot0)
        r_terminal = terminal_from_result if is_true_final_game_move else 0.0
        r = float(r_material + r_terminal)

        done = bool(is_true_final_game_move)
        terminal_reason = "pgn_result" if is_true_final_game_move else "playing"

        rec = {
            "phi": phi.tolist(),
            "r": r,
            "fen_next": fen_next,
            "done": done,
            "feature_version": feats.spec.version,
            "reward_version": REWARD_VERSION,
            "source": "pgn",
        }

        if keep_metadata:
            rec.update(
                {
                    "event": game.headers.get("Event", ""),
                    "site": game.headers.get("Site", ""),
                    "date": game.headers.get("Date", ""),
                    "round": game.headers.get("Round", ""),
                    "white": game.headers.get("White", ""),
                    "black": game.headers.get("Black", ""),
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "result": result,
                    "ply_index": ply_index,
                    "uci": uci,
                    "fen_before": fen_before,
                    "white_to_move_before": bool(white_to_move_before),
                    "material_before": float(pot0),
                    "material_after": float(pot1),
                    "material_delta": float(pot1 - pot0),
                    "reward_material": float(r_material),
                    "reward_terminal": float(r_terminal),
                    "terminal_reason": terminal_reason,
                }
            )

        out.append(rec)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="v1_basic")
    ap.add_argument("--reward-alpha", type=float, default=0.05)
    ap.add_argument("--result-scale", type=float, default=1.0)
    ap.add_argument("--min-elo", type=int, default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=None)

    ap.add_argument(
        "--keep-metadata",
        action="store_true",
        help=(
            "Store debug/provenance metadata for every sample. "
            "Useful for audits, slower and larger output."
        ),
    )

    ap.add_argument(
        "--compresslevel",
        type=int,
        default=1,
        help="gzip compression level for output. 1 is fastest, 9 is smallest.",
    )

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feats = get_features(args.features)

    games_seen = 0
    games_used = 0
    games_skipped = 0
    samples_written = 0

    t0 = time.time()

    with open_pgn_text(args.pgn) as pgn_f, gzip.open(
        out_path,
        "wt",
        encoding="utf-8",
        compresslevel=args.compresslevel,
    ) as out_f:
        if args.max_samples is not None:
            pbar = tqdm(
                total=args.max_samples,
                unit="samples",
                desc="PGN samples",
                mininterval=0.5,
            )
        else:
            pbar = tqdm(
                unit="game",
                desc="PGN games",
                mininterval=0.5,
            )

        last_postfix_t = 0.0

        def update_postfix(*, force: bool = False) -> None:
            nonlocal last_postfix_t

            now = time.time()
            if not force and (now - last_postfix_t) < 0.5:
                return

            last_postfix_t = now
            dt_now = max(1e-9, now - t0)

            pbar.set_postfix_str(
                f"games={games_seen} "
                f"used={games_used} "
                f"skip={games_skipped}"
            )

        while True:
            if args.max_games is not None and games_seen >= args.max_games:
                break

            if args.max_samples is not None and samples_written >= args.max_samples:
                break

            game = chess.pgn.read_game(pgn_f)
            if game is None:
                break

            games_seen += 1

            recs = process_game(
                game,
                feats=feats,
                reward_alpha=args.reward_alpha,
                result_scale=args.result_scale,
                min_elo=args.min_elo,
                max_plies=args.max_plies,
                keep_metadata=args.keep_metadata,
            )

            if not recs:
                games_skipped += 1

                if args.max_samples is None:
                    pbar.update(1)

                update_postfix()
                continue

            games_used += 1

            for rec in recs:
                if args.max_samples is not None and samples_written >= args.max_samples:
                    break

                out_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                samples_written += 1

                if args.max_samples is not None:
                    pbar.update(1)

            if args.max_samples is None:
                pbar.update(1)

            update_postfix()

        update_postfix(force=True)
        pbar.close()

    dt = max(1e-9, time.time() - t0)

    print(f"wrote: {out_path}")
    print(f"games seen: {games_seen}")
    print(f"games used: {games_used}")
    print(f"games skipped: {games_skipped}")
    print(f"samples written: {samples_written}")
    print(f"time: {dt:.1f}s")
    print(f"games/sec: {games_seen / dt:.1f}")
    print(f"samples/sec: {samples_written / dt:.1f}")


if __name__ == "__main__":
    main()