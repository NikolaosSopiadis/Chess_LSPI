from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FEATURE_GROUPS: dict[str, list[int]] = {
    # Old/core groups
    "basic_material": [1, 2, 3, 4, 5],
    "mobility": [9],
    "basic_tactics": [10, 11, 12],
    "pawns": [13, 14, 15],
    "terminal_draw": [16, 17, 18, 19, 20, 21, 22, 23],
    "king_safety_old": list(range(24, 37)),

    # v6 attack-map group
    "attackmap_all": list(range(37, 57)),
    "attackmap_global": [37, 38, 39, 40, 41, 42, 43, 44],
    "attackmap_kingzone": [45, 46, 47, 48, 49],
    "queen_pressure": [50, 51, 52],
    "loose_overload": [53, 54, 55, 56],

    # v7 API-tactics group
    "api_tactics_all": list(range(57, 64)),
    "checking_api": [57, 58, 59],
    "capture_api": [60, 61],
    "queen_tempo_api": [62, 63],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(ckpt_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    w = np.array(payload["w"], dtype=np.float64, copy=True)

    for group_name in args.group:
        if group_name not in FEATURE_GROUPS:
            known = ", ".join(sorted(FEATURE_GROUPS))
            raise ValueError(f"unknown group {group_name!r}; known groups: {known}")

        w2 = w.copy()
        indices = FEATURE_GROUPS[group_name]

        for idx in indices:
            if not (0 <= idx < len(w2)):
                raise IndexError(
                    f"group {group_name!r} contains index {idx}, "
                    f"but checkpoint has dim {len(w2)}"
                )

        w2[indices] = 0.0

        out_payload = dict(payload)
        out_payload["w"] = w2

        out_path = out_dir / f"{ckpt_path.stem}.no_{group_name}.npz"
        np.savez(out_path, **out_payload)

        print(f"wrote: {out_path}")
        print(f"  zeroed: {group_name}")
        print(f"  indices: {indices}")


if __name__ == "__main__":
    main()