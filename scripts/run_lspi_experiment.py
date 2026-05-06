from __future__ import annotations

import argparse
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


# -----------------------------
# Constants
# -----------------------------

PGN_PATH = Path("data/raw/lichess/games/lichess_db_standard_rated_2025-12.pgn.zst")

DEFAULT_MIN_ELO = 2000
SEED = 1

SHARDS = 12
WORKERS = 12

ITERS = 10
REG = "1e-1"
GAMMA = "0.99"

MAX_PLIES = 250
RANDOM_OPENINGS = 6

RANDOM_EVAL_STARTS = 50      # with --swap-colors => 100 games
MATERIAL_EVAL_STARTS = 100   # with --swap-colors => 200 games

MATRIX_GAMES = 40
MATRIX_WORKERS = 11
MATRIX_CHUNK_SIZE = 2
MATRIX_SEED = 10

DATA_DIR = Path("data/processed/samples")
CKPT_DIR = Path("data/processed/checkpoints")
EVAL_DIR = Path("data/processed/eval")


# -----------------------------
# Helpers
# -----------------------------

def sample_label(n: int) -> str:
    if n == 0:
        return "0"
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def normalize_agent_name(name: str) -> str:
    """
    User-facing aliases:

      v2_1       -> v2_1_basic
      v3         -> v3_basic
      v4         -> v4_slim
      v5         -> v5_center

    Already-expanded names pass through:

      v3_basic
      v4_slim
      v5_center
    """
    name = name.strip()

    aliases = {
        "v1": "v1_basic",
        "v1_basic": "v1_basic",

        "v2": "v2_basic",
        "v2_basic": "v2_basic",

        "v2_1": "v2_1_basic",
        "v2_1_basic": "v2_1_basic",

        "v3": "v3_basic",
        "v3_basic": "v3_basic",

        "v4": "v4_slim",
        "v4_slim": "v4_slim",

        "v5": "v5_center",
        "v5_center": "v5_center",

        "v6": "v6_attackmap",
        "v6_attackmap": "v6_attackmap",
    }

    if name in aliases:
        return aliases[name]

    if name.endswith(("_basic", "_slim", "_center")):
        return name

    if name.startswith("v"):
        return f"{name}_basic"

    return name


def dataset_feature_label(feature_name: str) -> str:
    """
    For file names:

      v3_basic   -> v3
      v2_1_basic -> v2_1
      v4_slim    -> v4_slim
      v5_center  -> v5_center
    """
    if feature_name.endswith("_basic"):
        return feature_name[: -len("_basic")]
    return feature_name


def safe_label(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def remove_file(path: Path) -> None:
    if path.exists():
        print(f"[remove] {path}")
        path.unlink()


def remove_dir(path: Path) -> None:
    if path.exists():
        print(f"[remove] {path}")
        shutil.rmtree(path)


def remove_checkpoint_family(path: Path) -> None:
    remove_file(path)

    pattern = f"{path.stem}.iter*.npz"
    for p in path.parent.glob(pattern):
        print(f"[remove] {p}")
        p.unlink()


def shards_exist(shard_dir: Path, expected: int) -> bool:
    if not shard_dir.exists():
        return False
    return len(list(shard_dir.glob("*.jsonl.gz"))) >= expected


def run_step(desc: str, cmd: list[str], *, env: dict[str, str]) -> None:
    print()
    print("=" * 100)
    print(desc)
    print("=" * 100)
    print("$ " + shlex.join(cmd))
    print()

    subprocess.run(cmd, env=env, check=True)


def skip_step(desc: str, path: Path) -> None:
    print()
    print("=" * 100)
    print(f"SKIP: {desc}")
    print("=" * 100)
    print(f"Already exists: {path}")


def build_mix_name(
    *,
    pgn_label: str,
    anchor_label: str,
    center_label: str,
    ds_label: str,
    elo_label: str,
    seed: int,
) -> str:
    parts = [f"mix_pgn{pgn_label}"]

    if anchor_label != "0":
        parts.append(f"anchor{anchor_label}")

    if center_label != "0":
        parts.append(f"center{center_label}")

    parts.append(ds_label)
    parts.append(elo_label)
    parts.append(f"seed{seed}")

    return "_".join(parts)


def current_matrix_model_spec(
    *,
    label: str,
    kind: str,
    ckpt: Path,
    depth: int,
    max_branch: str,
    draw_safety: bool,
    tactical_safety: bool,
) -> str:
    if kind == "search":
        return (
            f"{label}:search:{ckpt}:"
            f"depth={depth},"
            f"max_branch={max_branch},"
            f"draw={int(draw_safety)},"
            f"tactical={int(tactical_safety)}"
        )

    return f"{label}:{kind}:{ckpt}"


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build/mix/split/train/evaluate an LSPI chess experiment."
    )

    ap.add_argument(
        "--agent",
        required=True,
        help="Feature/agent family, e.g. v3, v3_basic, v4, v5.",
    )
    ap.add_argument(
        "--pgn",
        required=True,
        type=int,
        help="Number of PGN samples to include in the final training mix.",
    )
    ap.add_argument(
        "--anchor",
        required=True,
        type=int,
        help="Number of material-anchor samples to include in the final training mix. Use 0 for PGN-only.",
    )
    ap.add_argument(
        "--center-anchor",
        type=int,
        default=0,
        help="Number of opening-center anchor samples to include in the final training mix.",
    )
    ap.add_argument(
        "--min-elo",
        type=int,
        default=DEFAULT_MIN_ELO,
        help="Minimum Elo for PGN game sampling.",
    )

    ap.add_argument(
        "--center-bonus-scale",
        type=float,
        default=1.0,
        help="Opening-center anchor bonus scale.",
    )
    ap.add_argument(
        "--center-bonus-clip",
        type=float,
        default=0.35,
        help="Opening-center anchor bonus clip.",
    )
    ap.add_argument(
        "--center-target-clip",
        type=float,
        default=4.0,
        help="Opening-center anchor final target clip.",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="Regenerate data, shards, and checkpoint even if they already exist.",
    )

    # Evaluation controls.
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only build/train. Do not inspect or run any evaluation.",
    )
    ap.add_argument(
        "--no-inspect",
        action="store_true",
        help="Do not run inspect_checkpoint after training.",
    )
    ap.add_argument(
        "--eval-position-suite",
        action="store_true",
        help="Run eval_position_suite after training. Default: off.",
    )
    ap.add_argument(
        "--eval-random",
        action="store_true",
        help="Run tactical eval vs random after training. Default: off.",
    )
    ap.add_argument(
        "--eval-material",
        action="store_true",
        help="Run tactical eval vs material/greedy after training. Default: off.",
    )

    # Matrix evaluation.
    ap.add_argument(
        "--matrix-model",
        action="append",
        default=[],
        help=(
            "Extra model spec for scripts.eval_model_matrix. "
            "Example: v4.2:search:data/processed/checkpoints/foo.npz:"
            "depth=2,max_branch=none,draw=1,tactical=1. "
            "Can be passed multiple times. If omitted, matrix eval is skipped."
        ),
    )
    ap.add_argument(
        "--matrix-kind",
        choices=["plain", "tactical", "search"],
        default="search",
        help="Wrapper used for the newly trained model in matrix eval.",
    )
    ap.add_argument(
        "--matrix-label",
        default=None,
        help="Label for the newly trained model in matrix eval. Default: auto.",
    )
    ap.add_argument("--matrix-games", type=int, default=MATRIX_GAMES)
    ap.add_argument("--matrix-workers", type=int, default=MATRIX_WORKERS)
    ap.add_argument("--matrix-chunk-size", type=int, default=MATRIX_CHUNK_SIZE)
    ap.add_argument("--matrix-seed", type=int, default=MATRIX_SEED)
    ap.add_argument(
        "--matrix-position-source",
        choices=["suite", "random"],
        default="suite",
    )
    ap.add_argument("--matrix-random-openings", type=int, default=RANDOM_OPENINGS)
    ap.add_argument("--matrix-depth", type=int, default=2)
    ap.add_argument("--matrix-max-branch", default="none")
    ap.add_argument("--matrix-no-draw-safety", action="store_true")
    ap.add_argument("--matrix-no-tactical-safety", action="store_true")
    ap.add_argument(
        "--matrix-json-out",
        default=None,
        help="Optional JSON output path for matrix eval. Default: auto when matrix eval runs.",
    )

    args = ap.parse_args()

    feature_name = normalize_agent_name(args.agent)
    ds_label = dataset_feature_label(feature_name)

    pgn_samples = args.pgn
    anchor_samples = args.anchor
    center_anchor_samples = args.center_anchor

    if pgn_samples <= 0:
        raise ValueError("--pgn must be positive")
    if anchor_samples < 0:
        raise ValueError("--anchor cannot be negative")
    if center_anchor_samples < 0:
        raise ValueError("--center-anchor cannot be negative")
    if args.min_elo <= 0:
        raise ValueError("--min-elo must be positive")

    # We build enough PGN rows for all sampled sources.
    source_pgn_samples = pgn_samples + anchor_samples + center_anchor_samples

    pgn_label = sample_label(pgn_samples)
    anchor_label = sample_label(anchor_samples)
    center_label = sample_label(center_anchor_samples)
    source_label = sample_label(source_pgn_samples)
    elo_label = f"{args.min_elo}elo"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    pgn_src = DATA_DIR / f"pgn_{source_label}_{ds_label}_{elo_label}_seed{SEED}.jsonl.gz"

    anchor_src: Path | None = None
    if anchor_samples > 0:
        anchor_src = (
            DATA_DIR
            / f"material_anchor_{anchor_label}_from_pgn_{source_label}_{ds_label}_{elo_label}_seed{SEED}.jsonl.gz"
        )

    center_anchor_src: Path | None = None
    if center_anchor_samples > 0:
        center_anchor_src = (
            DATA_DIR
            / f"opening_center_anchor_{center_label}_from_pgn_{source_label}_{ds_label}_{elo_label}_seed{SEED}.jsonl.gz"
        )

    if anchor_samples > 0 or center_anchor_samples > 0:
        mix_name = build_mix_name(
            pgn_label=pgn_label,
            anchor_label=anchor_label,
            center_label=center_label,
            ds_label=ds_label,
            elo_label=elo_label,
            seed=SEED,
        )

        train_src = DATA_DIR / f"{mix_name}.jsonl.gz"
        shard_dir = DATA_DIR / f"{mix_name}_shards"

        ckpt_parts = [f"lspi_{feature_name}", "mix", f"pgn{pgn_label}"]

        if anchor_samples > 0:
            ckpt_parts.append(f"anchor{anchor_label}")

        if center_anchor_samples > 0:
            ckpt_parts.append(f"center{center_label}")

        ckpt_parts.append(elo_label)
        ckpt_parts.append(f"reg{REG}")

        ckpt = CKPT_DIR / ("_".join(ckpt_parts) + ".npz")
        exp_id = "_".join(ckpt_parts[:-1] + [f"seed{SEED}"])
    else:
        train_src = pgn_src
        shard_dir = DATA_DIR / f"pgn_{source_label}_{ds_label}_{elo_label}_seed{SEED}_shards"
        ckpt = CKPT_DIR / f"lspi_{feature_name}_pgn{pgn_label}_{elo_label}_reg{REG}.npz"
        exp_id = f"lspi_{feature_name}_pgn{pgn_label}_{elo_label}_seed{SEED}"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    py = sys.executable

    if args.force:
        remove_file(pgn_src)

        if anchor_src is not None:
            remove_file(anchor_src)

        if center_anchor_src is not None:
            remove_file(center_anchor_src)

        if train_src != pgn_src:
            remove_file(train_src)
            remove_file(Path(str(train_src) + ".manifest.json"))

        remove_dir(shard_dir)
        remove_checkpoint_family(ckpt)

    print()
    print("=" * 100)
    print("EXPERIMENT")
    print("=" * 100)
    print(f"Experiment:        {exp_id}")
    print(f"Feature:           {feature_name}")
    print(f"Dataset label:     {ds_label}")
    print(f"Min Elo:           {args.min_elo}")
    print(f"PGN rows:          {pgn_samples}")
    print(f"Material anchors:  {anchor_samples}")
    print(f"Center anchors:    {center_anchor_samples}")
    print(f"Source PGN rows:   {source_pgn_samples}")
    print(f"PGN source:        {pgn_src}")
    print(f"Material source:   {anchor_src}")
    print(f"Center source:     {center_anchor_src}")
    print(f"Training dataset:  {train_src}")
    print(f"Shard dir:         {shard_dir}")
    print(f"Checkpoint:        {ckpt}")

    # 1. Build PGN source dataset.
    if pgn_src.exists() and not args.force:
        skip_step("build PGN source dataset", pgn_src)
    else:
        run_step(
            "build PGN source dataset",
            [
                py,
                "-m",
                "scripts.build_dataset_pgn",
                "--pgn",
                str(PGN_PATH),
                "--out",
                str(pgn_src),
                "--features",
                feature_name,
                "--min-elo",
                str(args.min_elo),
                "--max-samples",
                str(source_pgn_samples),
            ],
            env=env,
        )

    # 2. Build material anchors.
    if anchor_samples > 0:
        assert anchor_src is not None

        if anchor_src.exists() and not args.force:
            skip_step("build material anchors", anchor_src)
        else:
            run_step(
                "build material anchors",
                [
                    py,
                    "-m",
                    "scripts.build_dataset_material_anchor",
                    "--src",
                    str(pgn_src),
                    "--out",
                    str(anchor_src),
                    "--features",
                    feature_name,
                    "--max-samples",
                    str(anchor_samples),
                    "--seed",
                    str(SEED),
                    "--scale",
                    "1.0",
                    "--clip",
                    "4.0",
                ],
                env=env,
            )

    # 3. Build opening-center anchors.
    if center_anchor_samples > 0:
        assert center_anchor_src is not None

        if center_anchor_src.exists() and not args.force:
            skip_step("build opening-center anchors", center_anchor_src)
        else:
            run_step(
                "build opening-center anchors",
                [
                    py,
                    "-m",
                    "scripts.build_dataset_opening_center_anchor",
                    "--src",
                    str(pgn_src),
                    "--out",
                    str(center_anchor_src),
                    "--features",
                    feature_name,
                    "--max-samples",
                    str(center_anchor_samples),
                    "--seed",
                    str(SEED),
                    "--material-scale",
                    "1.0",
                    "--bonus-scale",
                    str(args.center_bonus_scale),
                    "--bonus-clip",
                    str(args.center_bonus_clip),
                    "--target-clip",
                    str(args.center_target_clip),
                ],
                env=env,
            )

    # 4. Mix dataset.
    if anchor_samples > 0 or center_anchor_samples > 0:
        if train_src.exists() and not args.force:
            skip_step("mix dataset", train_src)
        else:
            cmd = [
                py,
                "-m",
                "scripts.mix_datasets",
                "--src",
                f"{pgn_src}:{pgn_samples}",
            ]

            if anchor_samples > 0:
                assert anchor_src is not None
                cmd.extend(["--src", f"{anchor_src}:{anchor_samples}"])

            if center_anchor_samples > 0:
                assert center_anchor_src is not None
                cmd.extend(["--src", f"{center_anchor_src}:{center_anchor_samples}"])

            cmd.extend(
                [
                    "--out",
                    str(train_src),
                    "--seed",
                    str(SEED),
                ]
            )

            run_step("mix dataset", cmd, env=env)

    # 5. Split dataset.
    if shards_exist(shard_dir, SHARDS) and not args.force:
        skip_step("split dataset", shard_dir)
    else:
        run_step(
            "split dataset",
            [
                py,
                "-m",
                "scripts.split_dataset",
                "--src",
                str(train_src),
                "--out-dir",
                str(shard_dir),
                "--shards",
                str(SHARDS),
            ],
            env=env,
        )

    # 6. Train.
    if ckpt.exists() and not args.force:
        skip_step("train LSPI", ckpt)
    else:
        run_step(
            "train LSPI",
            [
                py,
                "-m",
                "scripts.train_lspi",
                "--samples",
                str(train_src),
                "--shards-dir",
                str(shard_dir),
                "--out",
                str(ckpt),
                "--features",
                feature_name,
                "--workers",
                str(WORKERS),
                "--preload",
                "--iters",
                str(ITERS),
                "--reg",
                REG,
                "--gamma",
                GAMMA,
                "--ckpt-every-iter",
            ],
            env=env,
        )

    # 7. Optional evaluation.
    if not args.skip_eval:
        if not args.no_inspect:
            run_step(
                "inspect checkpoint",
                [
                    py,
                    "-m",
                    "scripts.inspect_checkpoint",
                    str(ckpt),
                ],
                env=env,
            )

        if args.eval_position_suite:
            run_step(
                "position suite",
                [
                    py,
                    "-m",
                    "scripts.eval_position_suite",
                    "--agent",
                    "lspi_v1",
                    "--path",
                    str(ckpt),
                    "--show-top",
                    "10",
                ],
                env=env,
            )

        if args.eval_random:
            run_step(
                "self-play eval vs random",
                [
                    py,
                    "-m",
                    "scripts.eval_selfplay",
                    "--white",
                    "lspi_tactical",
                    "--black",
                    "random",
                    "--lspi-v1-path",
                    str(ckpt),
                    "--games",
                    str(RANDOM_EVAL_STARTS),
                    "--max-plies",
                    str(MAX_PLIES),
                    "--random-openings",
                    str(RANDOM_OPENINGS),
                    "--swap-colors",
                    "--seed",
                    "3",
                ],
                env=env,
            )

        if args.eval_material:
            run_step(
                "self-play eval vs material",
                [
                    py,
                    "-m",
                    "scripts.eval_selfplay",
                    "--white",
                    "lspi_tactical",
                    "--black",
                    "material",
                    "--lspi-v1-path",
                    str(ckpt),
                    "--games",
                    str(MATERIAL_EVAL_STARTS),
                    "--max-plies",
                    str(MAX_PLIES),
                    "--random-openings",
                    str(RANDOM_OPENINGS),
                    "--swap-colors",
                    "--seed",
                    "2",
                ],
                env=env,
            )

        if args.matrix_model:
            matrix_draw_safety = not args.matrix_no_draw_safety
            matrix_tactical_safety = not args.matrix_no_tactical_safety

            if args.matrix_label is not None:
                trained_label = args.matrix_label
            else:
                tag_parts = [ds_label, f"pgn{pgn_label}"]

                if anchor_samples > 0:
                    tag_parts.append(f"anchor{anchor_label}")

                if center_anchor_samples > 0:
                    tag_parts.append(f"center{center_label}")

                tag_parts.append(elo_label)
                tag_parts.append(args.matrix_kind)

                trained_label = safe_label("_".join(tag_parts))

            trained_spec = current_matrix_model_spec(
                label=trained_label,
                kind=args.matrix_kind,
                ckpt=ckpt,
                depth=args.matrix_depth,
                max_branch=args.matrix_max_branch,
                draw_safety=matrix_draw_safety,
                tactical_safety=matrix_tactical_safety,
            )

            if args.matrix_json_out is not None:
                matrix_json_out = Path(args.matrix_json_out)
            else:
                matrix_json_out = (
                    EVAL_DIR
                    / f"model_matrix_{trained_label}_seed{args.matrix_seed}.json"
                )

            cmd = [
                py,
                "-m",
                "scripts.eval_model_matrix",
                "--model",
                trained_spec,
            ]

            for model_spec in args.matrix_model:
                cmd.extend(["--model", model_spec])

            cmd.extend(
                [
                    "--games",
                    str(args.matrix_games),
                    "--position-source",
                    args.matrix_position_source,
                    "--max-plies",
                    str(MAX_PLIES),
                    "--random-openings",
                    str(args.matrix_random_openings),
                    "--seed",
                    str(args.matrix_seed),
                    "--workers",
                    str(args.matrix_workers),
                    "--chunk-size",
                    str(args.matrix_chunk_size),
                    "--json-out",
                    str(matrix_json_out),
                ]
            )

            run_step("model matrix evaluation", cmd, env=env)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()