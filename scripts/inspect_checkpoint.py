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
    
    "v4_slim": [
        "bias",

        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",

        "side_to_move_pm",
        "white_in_check",
        "black_in_check",

        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",

        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",

        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",

        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",

        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",
    ],

    "v5_center": [
        # v4_slim base
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",
        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",

        # v5 center features
        "center_pawn_presence_diff",
        "extended_center_pawn_presence_diff",
        "center_piece_occupation_diff",
        "center_control_diff",
        "extended_center_control_diff",
        "center_minor_control_diff",
        "center_pawn_control_diff",
        "opening_center_control_diff",
        "opening_center_pawn_presence_diff",
        "queen_out_before_center_diff",
        "white_queen_out_before_center",
        "black_queen_out_before_center",
    ],
    
    "v6_attackmap": [
        # v4_slim base
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",
        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",

        # v6 attack-map features
        "attack_coverage_diff",
        "attack_volume_diff",
        "minor_attack_volume_diff",
        "pawn_attack_volume_diff",
        "nonqueen_attack_volume_diff",
        "center_attack_volume_diff",
        "extended_center_attack_volume_diff",
        "own_piece_defense_volume_diff",
        "king_zone_attack_volume_diff",
        "king_zone_multi_attack_diff",
        "king_zone_pawn_attack_diff",
        "king_zone_minor_attack_diff",
        "king_zone_net_attack_diff",
        "queen_pressure_diff",
        "queen_pressure_by_minor_or_pawn_diff",
        "queen_net_attack_defense_diff",
        "loose_piece_pressure_diff",
        "overloaded_piece_pressure_diff",
        "multi_attacked_material_diff",
        "tempo_liability_diff",
    ],
    
    "v7_api_tactics": [
        # v6
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",
        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",

        # v6 attack-map features
        "attack_coverage_diff",
        "attack_volume_diff",
        "minor_attack_volume_diff",
        "pawn_attack_volume_diff",
        "nonqueen_attack_volume_diff",
        "center_attack_volume_diff",
        "extended_center_attack_volume_diff",
        "own_piece_defense_volume_diff",
        "king_zone_attack_volume_diff",
        "king_zone_multi_attack_diff",
        "king_zone_pawn_attack_diff",
        "king_zone_minor_attack_diff",
        "king_zone_net_attack_diff",
        "queen_pressure_diff",
        "queen_pressure_by_minor_or_pawn_diff",
        "queen_net_attack_defense_diff",
        "loose_piece_pressure_diff",
        "overloaded_piece_pressure_diff",
        "multi_attacked_material_diff",
        "tempo_liability_diff",
        
        # v7
        "legal_checking_moves_diff",
        "safe_checking_moves_diff",
        "mate_in_one_threat_diff",
        "safe_capture_value_diff",
        "unsafe_capture_liability_diff",
        "queen_tempo_threat_diff",
        "queen_tempo_threat_by_minor_or_pawn_diff",
    ],
    
    "v8_api_tactics_clean": [
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",
        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",

        # v8 additions
        "legal_checking_moves_diff",
        "safe_checking_moves_diff",
        "mate_in_one_threat_diff",
        "safe_capture_value_diff",
        "unsafe_capture_liability_diff",
        "queen_tempo_threat_diff",
        "queen_tempo_threat_by_minor_or_pawn_diff",
    ],
    
    "v9_response_tactics": [
        # 0..43 = v8_api_tactics_clean
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",
        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",
        "legal_checking_moves_diff",
        "safe_checking_moves_diff",
        "mate_in_one_threat_diff",
        "safe_capture_value_diff",
        "unsafe_capture_liability_diff",
        "queen_tempo_threat_diff",
        "queen_tempo_threat_by_minor_or_pawn_diff",

        # 44..57 concrete components behind v8 tactical diffs
        "white_legal_checking_moves",
        "black_legal_checking_moves",
        "white_safe_checking_moves",
        "black_safe_checking_moves",
        "white_mate_in_one_moves",
        "black_mate_in_one_moves",
        "white_safe_capture_value",
        "black_safe_capture_value",
        "white_best_safe_capture_value",
        "black_best_safe_capture_value",
        "white_unsafe_capture_liability",
        "black_unsafe_capture_liability",
        "white_queen_tempo_minor_or_pawn",
        "black_queen_tempo_minor_or_pawn",

        # 58..62 side-to-move immediate response features
        "stm_safe_capture_value_pm",
        "stm_best_safe_capture_value_pm",
        "stm_safe_checking_moves_pm",
        "stm_mate_in_one_moves_pm",
        "stm_queen_tempo_minor_or_pawn_pm",

        # 63..67 not-side-to-move latent threat features
        "ntm_safe_capture_value_pm",
        "ntm_best_safe_capture_value_pm",
        "ntm_safe_checking_moves_pm",
        "ntm_mate_in_one_moves_pm",
        "ntm_queen_tempo_minor_or_pawn_pm",

        # 68..79 concrete opening / queen exposure features
        "white_queen_exposure_opening",
        "black_queen_exposure_opening",
        "queen_exposure_liability_diff",
        "white_unfinished_development_opening",
        "black_unfinished_development_opening",
        "development_deficit_diff",
        "white_king_center_opening",
        "black_king_center_opening",
        "king_center_liability_diff",
        "white_castle_ready_opening",
        "black_castle_ready_opening",
        "castle_readiness_diff",
    ],
    
    "v10_response_fast": [
        "bias",
        "pawn_diff",
        "knight_diff",
        "bishop_diff",
        "rook_diff",
        "queen_diff",
        "side_to_move_pm",
        "white_in_check",
        "black_in_check",
        "mobility_diff",
        "attacked_material_diff",
        "hanging_material_diff",
        "king_pressure_diff",
        "pawn_advancement_diff",
        "passed_pawn_diff",
        "promotion_pressure_diff",
        "terminal_checkmate_white_wins",
        "terminal_checkmate_black_wins",
        "white_ahead_draw_terminal",
        "black_ahead_draw_terminal",
        "white_ahead_repeat_risk",
        "black_ahead_repeat_risk",
        "halfmove_pressure_white_ahead",
        "halfmove_pressure_black_ahead",
        "castled_diff",
        "king_walked_uncastled_diff",
        "pawn_shield_diff",
        "king_zone_safety_diff",
        "open_file_safety_diff",
        "development_diff",
        "queen_development_diff",
        "white_castled",
        "black_castled",
        "white_pawn_shield",
        "black_pawn_shield",
        "white_king_zone_attacked",
        "black_king_zone_attacked",

        "white_legal_checking_moves",
        "black_legal_checking_moves",
        "white_safe_checking_moves",
        "black_safe_checking_moves",
        "white_mate_in_one_moves",
        "black_mate_in_one_moves",
        "white_safe_capture_value",
        "black_safe_capture_value",
        "white_best_safe_capture_value",
        "black_best_safe_capture_value",
        "stm_safe_capture_value_pm",
        "stm_best_safe_capture_value_pm",
        "stm_safe_checking_moves_pm",
        "stm_mate_in_one_moves_pm",
        "ntm_safe_capture_value_pm",
        "ntm_best_safe_capture_value_pm",
        "ntm_safe_checking_moves_pm",
        "ntm_mate_in_one_moves_pm",
    ]

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

    if len(w) == 37:
        return "v4_slim"

    if len(w) == 49:
        return "v5_center"

    if len(w) == 65:
        return "v6_attackmap"

    if len(w) == 64:
        return "v7_api_tactics"
    
    if len(w) == 44:
        return "v8_api_tactics_clean"

    if len(w) == 80:
        return "v9_response_tactics"

    if len(w) == 55:
        return "v10_response_fast"

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

    elif feature_version in (
        "v2_1_basic",
        "v3_basic",
        "v4_basic",
        "v4_slim",
        "v5_center",
        "v6_attackmap",
        "v7_api_tactics",
        "v8_api_tactics_clean",
        "v9_response_tactics",
        "v10_response_fast",
    ):
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
        help="Override feature version, e.g. v1_basic, v2_basic, v2_1_basic, v3_basic, v4_slim, v5_center, v7_api_tactics, v9_response_tactics.",
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