from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import random
from typing import Any, Iterator

from chess_core.board import Board
from chess_rl.rewards.v1_terminal_plus_potential import material_potential

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


FEATURE_VERSION = "v5_center"
REWARD_VERSION = "opening_center_anchor_v1"


# v5_center feature indices.
IDX_OPENING_CENTER_CONTROL_DIFF = 44
IDX_OPENING_CENTER_PAWN_PRESENCE_DIFF = 45
IDX_QUEEN_OUT_BEFORE_CENTER_DIFF = 46
IDX_WHITE_QUEEN_OUT_BEFORE_CENTER = 47
IDX_BLACK_QUEEN_OUT_BEFORE_CENTER = 48


def open_text_auto(path: str | Path, mode: str):
    path = Path(path)

    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")

    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text_auto(path, "rt") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            yield json.loads(line)


def reservoir_sample_rows(
    *,
    src: str | Path,
    max_samples: int,
    seed: int,
    feature_version: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Reservoir-sample eligible rows from a source JSONL/JSONL.GZ file.

    Eligible rows must have:
      - matching feature_version
      - phi
      - fen_next
    """
    rng = random.Random(seed)

    sample: list[dict[str, Any]] = []
    rows_seen = 0
    eligible_seen = 0

    iterator = iter_jsonl(src)

    if tqdm is not None:
        iterator = tqdm(iterator, desc="read source", unit="rows")

    for row in iterator:
        rows_seen += 1

        if row.get("feature_version") != feature_version:
            continue

        if "phi" not in row:
            continue

        if "fen_next" not in row:
            continue

        phi = row["phi"]

        if not isinstance(phi, list):
            continue

        if len(phi) <= IDX_BLACK_QUEEN_OUT_BEFORE_CENTER:
            continue

        eligible_seen += 1

        if len(sample) < max_samples:
            sample.append(row)
        else:
            j = rng.randrange(eligible_seen)
            if j < max_samples:
                sample[j] = row

    return sample, rows_seen, eligible_seen


def opening_center_bonus_from_phi(
    phi: list[float],
    *,
    bonus_scale: float,
    bonus_clip: float,
) -> float:
    """
    Compute a white-perspective opening-center bonus from v5_center features.

    Positive means good for White.
    Negative means good for Black.

    The key sign targets are:

      opening_center_control_diff       -> positive weight
      opening_center_pawn_presence_diff -> positive weight
      queen_out_before_center_diff      -> positive weight
      white_queen_out_before_center     -> negative weight
      black_queen_out_before_center     -> positive weight

    The queen terms are deliberately conservative. A fully bad early queen
    sortie should be worth a few pawns, not a whole queen.
    """
    opening_center_control_diff = float(phi[IDX_OPENING_CENTER_CONTROL_DIFF])
    opening_center_pawn_presence_diff = float(phi[IDX_OPENING_CENTER_PAWN_PRESENCE_DIFF])
    queen_out_before_center_diff = float(phi[IDX_QUEEN_OUT_BEFORE_CENTER_DIFF])
    white_queen_bad = float(phi[IDX_WHITE_QUEEN_OUT_BEFORE_CENTER])
    black_queen_bad = float(phi[IDX_BLACK_QUEEN_OUT_BEFORE_CENTER])

    bonus = 0.0

    # Center incentives.
    bonus += 0.08 * opening_center_control_diff
    bonus += 0.12 * opening_center_pawn_presence_diff

    # Queen discipline.
    #
    # queen_out_before_center_diff = black_bad - white_bad.
    #
    # Positive coefficient:
    #   black queen bad -> good for White
    #   white queen bad -> bad for White
    bonus += 0.15 * queen_out_before_center_diff

    # Extra direct pressure on the individual redundant features so LSPI does
    # not distribute the sign backwards across the correlated columns.
    bonus -= 0.075 * white_queen_bad
    bonus += 0.075 * black_queen_bad

    bonus *= bonus_scale

    if bonus > bonus_clip:
        bonus = bonus_clip
    elif bonus < -bonus_clip:
        bonus = -bonus_clip

    return float(bonus)


def build_anchor_row(
    row: dict[str, Any],
    *,
    feature_version: str,
    material_scale: float,
    bonus_scale: float,
    bonus_clip: float,
    target_clip: float,
) -> dict[str, Any]:
    phi = row["phi"]
    fen_next = row["fen_next"]

    board = Board()
    board.init_board(fen_next)

    material_value = float(material_potential(board)) * material_scale
    opening_bonus = opening_center_bonus_from_phi(
        phi,
        bonus_scale=bonus_scale,
        bonus_clip=bonus_clip,
    )

    target = material_value + opening_bonus

    if target > target_clip:
        target = target_clip
    elif target < -target_clip:
        target = -target_clip

    return {
        "phi": phi,
        "r": float(target),
        "done": True,
        "fen_next": fen_next,
        "feature_version": feature_version,
        "reward_version": REWARD_VERSION,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default=FEATURE_VERSION)
    ap.add_argument("--max-samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument(
        "--material-scale",
        type=float,
        default=1.0,
        help="Scale applied to material_potential(board_after).",
    )
    ap.add_argument(
        "--bonus-scale",
        type=float,
        default=1.0,
        help="Scale applied to the opening-center bonus.",
    )
    ap.add_argument(
        "--bonus-clip",
        type=float,
        default=0.35,
        help="Clip only the opening-center bonus to ±this value.",
    )
    ap.add_argument(
        "--target-clip",
        type=float,
        default=4.0,
        help="Clip final target value to ±this value.",
    )

    args = ap.parse_args()

    rows, rows_seen, eligible_seen = reservoir_sample_rows(
        src=args.src,
        max_samples=args.max_samples,
        seed=args.seed,
        feature_version=args.features,
    )

    if not rows:
        raise ValueError(
            f"No eligible rows found in {args.src!r} for feature_version={args.features!r}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0

    material_sum = 0.0
    bonus_sum = 0.0
    target_sum = 0.0
    bonus_min = float("inf")
    bonus_max = float("-inf")

    iterator = rows
    if tqdm is not None:
        iterator = tqdm(rows, desc="write opening-center anchors", unit="rows")

    with open_text_auto(out, "wt") as f:
        for row in iterator:
            anchor = build_anchor_row(
                row,
                feature_version=args.features,
                material_scale=args.material_scale,
                bonus_scale=args.bonus_scale,
                bonus_clip=args.bonus_clip,
                target_clip=args.target_clip,
            )

            # Stats only.
            phi = row["phi"]
            board = Board()
            board.init_board(row["fen_next"])

            material_value = float(material_potential(board)) * args.material_scale
            bonus = opening_center_bonus_from_phi(
                phi,
                bonus_scale=args.bonus_scale,
                bonus_clip=args.bonus_clip,
            )

            material_sum += material_value
            bonus_sum += bonus
            target_sum += float(anchor["r"])
            bonus_min = min(bonus_min, bonus)
            bonus_max = max(bonus_max, bonus)

            f.write(json.dumps(anchor, separators=(",", ":")) + "\n")
            n_written += 1

    print(f"wrote: {out}")
    print(f"source rows seen: {rows_seen}")
    print(f"eligible rows seen: {eligible_seen}")
    print(f"anchors written: {n_written}")
    print(f"material_scale: {args.material_scale}")
    print(f"bonus_scale: {args.bonus_scale}")
    print(f"bonus_clip: ±{args.bonus_clip}")
    print(f"target_clip: ±{args.target_clip}")

    if n_written:
        print()
        print("Anchor stats:")
        print(f"  avg material target component: {material_sum / n_written:+.6f}")
        print(f"  avg opening bonus:             {bonus_sum / n_written:+.6f}")
        print(f"  avg final target:              {target_sum / n_written:+.6f}")
        print(f"  opening bonus range:           {bonus_min:+.6f} to {bonus_max:+.6f}")


if __name__ == "__main__":
    main()