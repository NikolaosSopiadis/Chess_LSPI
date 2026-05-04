from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES: dict[str, list[str]] = {
    "v1_basic": [
        "bias",
        "material_total",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "white_K_castle",
        "white_Q_castle",
        "black_K_castle",
        "black_Q_castle",
        "side_to_move",
        "white_in_check",
        "black_in_check",
        "unused_14",
        "unused_15",
    ],

    "v2_basic": [
        "bias",
        "material_total",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "white_K_castle",
        "white_Q_castle",
        "black_K_castle",
        "black_Q_castle",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "white_mobility",
        "black_mobility",
        "attacked_material_diff",
        "white_attacks_black_material",
        "black_attacks_white_material",
        "hanging_material_diff",
        "white_hanging_material",
        "black_hanging_material",
        "king_pressure_diff",
        "white_king_danger",
        "black_king_danger",
        "pawn_advancement_diff",
        "white_pawn_advancement",
        "black_pawn_advancement",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "halfmove_clock",
    ],

    "v2_1_basic": [
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "white_K_castle",
        "white_Q_castle",
        "black_K_castle",
        "black_Q_castle",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "white_mobility",
        "black_mobility",
        "attacked_material_diff",
        "white_attacks_black_material",
        "black_attacks_white_material",
        "hanging_material_diff",
        "white_hanging_material",
        "black_hanging_material",
        "king_pressure_diff",
        "white_king_danger",
        "black_king_danger",
        "pawn_advancement_diff",
        "white_pawn_advancement",
        "black_pawn_advancement",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "halfmove_clock",
    ],
    
        "v3_basic": [
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "white_K_castle",
        "white_Q_castle",
        "black_K_castle",
        "black_Q_castle",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "white_mobility",
        "black_mobility",
        "attacked_material_diff",
        "white_attacks_black_material",
        "black_attacks_white_material",
        "hanging_material_diff",
        "white_hanging_material",
        "black_hanging_material",
        "king_pressure_diff",
        "white_king_danger",
        "black_king_danger",
        "pawn_advancement_diff",
        "white_pawn_advancement",
        "black_pawn_advancement",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "halfmove_clock",
        "white_legal_mobility",
        "black_legal_mobility",
        "legal_mobility_diff",
        "side_to_move_legal_mobility",
        "terminal_draw",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "repeat_count_norm",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "white_ahead_low_enemy_mobility",
        "black_ahead_low_enemy_mobility",
    ],
}


def load_meta(ckpt: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta" not in ckpt.files:
        return {}

    raw = ckpt["meta"]

    try:
        obj = raw.item()
    except Exception:
        return {"raw_meta": str(raw)}

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"raw_meta": obj}

    return {"raw_meta": repr(obj)}


def infer_feature_version(w: np.ndarray, meta: dict[str, Any]) -> str:
    for key in ("feature_name", "features", "feature_version"):
        value = meta.get(key)
        if isinstance(value, str) and value in FEATURE_NAMES:
            return value

    if len(w) == 16:
        return "v1_basic"

    if len(w) == 32:
        return "v2_basic"

    if len(w) == 31:
        return "v2_1_basic"

    if len(w) == 47:
        return "v3_basic"

    return "unknown"


def print_effective_piece_values(feature_version: str, w: np.ndarray) -> None:
    """
    Print approximate score contribution for gaining one extra piece.

    This is not centipawns. It is contribution to w·phi.
    """
    print()
    print("Effective material contributions:")
    print("  Units are model score contribution, not centipawns.")

    if feature_version in ("v1_basic", "v2_basic"):
        material_w = float(w[1])

        pawn = material_w * 0.100 + float(w[2]) / 8.0
        knight = material_w * 0.320 + float(w[3]) / 2.0
        bishop = material_w * 0.330 + float(w[4]) / 2.0
        rook = material_w * 0.500 + float(w[5]) / 2.0
        queen = material_w * 0.900 + float(w[6])

    elif feature_version == ("v2_1_basic", "v3_basic"):
        pawn = float(w[1]) / 8.0
        knight = float(w[2]) / 2.0
        bishop = float(w[3]) / 2.0
        rook = float(w[4]) / 2.0
        queen = float(w[5])

    else:
        print("  Cannot compute effective piece values for unknown feature version.")
        return

    vals = [
        ("pawn", pawn),
        ("knight", knight),
        ("bishop", bishop),
        ("rook", rook),
        ("queen", queen),
    ]

    for name, val in vals:
        if abs(pawn) > 1e-12:
            ratio = val / pawn
            print(f"  {name:7s}: {val:+.6f}   ratio_to_pawn={ratio:+.2f}")
        else:
            print(f"  {name:7s}: {val:+.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument(
        "--features",
        default=None,
        help="Override feature version, e.g. v1_basic, v2_basic, v2_1_basic.",
    )
    args = ap.parse_args()

    ckpt = np.load(args.checkpoint, allow_pickle=True)

    print(f"checkpoint: {args.checkpoint}")
    print(f"files: {ckpt.files}")

    w = np.asarray(ckpt["w"], dtype=np.float64)
    meta = load_meta(ckpt)

    feature_version = args.features or infer_feature_version(w, meta)

    print()
    print(f"feature version: {feature_version}")
    print(f"dim: {len(w)}")

    if meta:
        print()
        print("meta:")
        for k, v in sorted(meta.items()):
            print(f"  {k}: {v}")

    names = FEATURE_NAMES.get(feature_version)

    if names is None:
        names = [f"feature_{i}" for i in range(len(w))]

    if len(names) != len(w):
        print()
        print(
            f"Warning: name count {len(names)} does not match weight count {len(w)}. "
            "Falling back to generic names."
        )
        names = [f"feature_{i}" for i in range(len(w))]

    print()
    print("weights:")
    for i, (name, val) in enumerate(zip(names, w)):
        print(f"{i:2d} {name:32s} {val:+.6f}")

    print_effective_piece_values(feature_version, w)


if __name__ == "__main__":
    main()