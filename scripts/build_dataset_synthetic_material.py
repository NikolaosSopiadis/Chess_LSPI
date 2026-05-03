from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from chess_core.board import Board
from chess_rl.features.registry import get as get_features
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


CASES = [
    # White captures
    "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",  # white pawn can take queen
    "4k3/8/8/7q/8/8/8/3QK3 w - - 0 1",    # white queen can take queen
    "4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1",  # white pawn can take rook
    "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",  # white pawn can take pawn

    # Black captures
    "4k3/8/8/4p3/3Q4/8/8/4K3 b - - 0 1",  # black pawn can take queen
    "3qk3/8/8/8/7Q/8/8/4K3 b - - 0 1",    # black queen can take queen
    "4k3/8/8/4p3/3R4/8/8/4K3 b - - 0 1",  # black pawn can take rook
    "4k3/8/8/4p3/3P4/8/8/4K3 b - - 0 1",  # black pawn can take pawn
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="v1_basic")
    ap.add_argument("--repeat", type=int, default=1000)
    args = ap.parse_args()

    feats = get_features(args.features)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0

    with gzip.open(out, "wt", encoding="utf-8") as f:
        for _ in range(args.repeat):
            for fen in CASES:
                b = Board()
                b.init_board(fen)

                pot0 = material_potential(b)

                for move in b.get_all_legal_moves():
                    phi = feats.phi_sa(b, move)

                    undo = b._do_move(move)
                    try:
                        pot1 = material_potential(b)
                        done, reason = b.game_end_state()
                        fen_next = b.to_fen()
                    finally:
                        b._undo_move(undo)

                    r = float(pot1 - pot0)

                    rec = {
                        "phi": phi.tolist(),
                        "r": r,
                        "fen_next": fen_next,
                        "done": bool(done),

                        "feature_version": feats.spec.version,
                        "reward_version": "synthetic_material_delta_v1",
                        "source": "synthetic_material",

                        "fen_before": fen,
                        "white_to_move_before": bool(b.get_is_white_to_move()),

                        "move_src": b.idx_to_algebraic(move.src_square),
                        "move_dst": b.idx_to_algebraic(move.dst_square),
                        "promotion": int(move.promotion),

                        "material_before": float(pot0),
                        "material_after": float(pot1),
                        "material_delta": float(pot1 - pot0),
                        "terminal_reason": reason,
                    }

                    f.write(json.dumps(rec) + "\n")
                    n += 1

    print(f"wrote {n} samples to {out}")


if __name__ == "__main__":
    main()