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

MIN_ELO = 1800
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

DATA_DIR = Path("data/processed/samples")
CKPT_DIR = Path("data/processed/checkpoints")


# -----------------------------
# Helpers
# -----------------------------

def sample_label(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def normalize_agent_name(name: str) -> str:
    """
    User-facing:
      v3       -> v3_basic
      v2_1     -> v2_1_basic
      v3_basic -> v3_basic
    """
    name = name.strip()

    if name.endswith("_basic"):
        return name

    if name.endswith("_slim"):
        return name
    
    if name.startswith("v4"):
        return f"{name}_slim"

    if name.startswith("v"):
        return f"{name}_basic"

    return name


def dataset_feature_label(feature_name: str) -> str:
    """
    For file names:
      v3_basic   -> v3
      v2_1_basic -> v2_1
    """
    if feature_name.endswith("_basic"):
        return feature_name[: -len("_basic")]
    return feature_name


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
        help="Feature/agent family, e.g. v3, v3_basic, v2_1, v2_1_basic.",
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
        "--force",
        action="store_true",
        help="Regenerate data, shards, and checkpoint even if they already exist.",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only build/train. Do not run inspect/position/random/material evaluations.",
    )

    args = ap.parse_args()

    feature_name = normalize_agent_name(args.agent)
    ds_label = dataset_feature_label(feature_name)

    pgn_samples = args.pgn
    anchor_samples = args.anchor

    if pgn_samples <= 0:
        raise ValueError("--pgn must be positive")
    if anchor_samples < 0:
        raise ValueError("--anchor cannot be negative")

    # We build enough PGN rows for both the PGN training subset and the anchor source.
    source_pgn_samples = pgn_samples + anchor_samples

    pgn_label = sample_label(pgn_samples)
    anchor_label = sample_label(anchor_samples)
    source_label = sample_label(source_pgn_samples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    pgn_src = DATA_DIR / f"pgn_{source_label}_{ds_label}_seed{SEED}.jsonl.gz"

    if anchor_samples > 0:
        anchor_src = (
            DATA_DIR
            / f"material_anchor_{anchor_label}_from_pgn_{source_label}_{ds_label}_seed{SEED}.jsonl.gz"
        )

        train_src = (
            DATA_DIR
            / f"mix_pgn{pgn_label}_anchor{anchor_label}_{ds_label}_seed{SEED}.jsonl.gz"
        )

        shard_dir = (
            DATA_DIR
            / f"mix_pgn{pgn_label}_anchor{anchor_label}_{ds_label}_seed{SEED}_shards"
        )

        ckpt = (
            CKPT_DIR
            / f"lspi_{feature_name}_mix_pgn{pgn_label}_anchor{anchor_label}_reg{REG}.npz"
        )

        exp_id = f"lspi_{feature_name}_mix_pgn{pgn_label}_anchor{anchor_label}_seed{SEED}"
    else:
        anchor_src = None
        train_src = pgn_src

        shard_dir = DATA_DIR / f"pgn_{source_label}_{ds_label}_seed{SEED}_shards"

        ckpt = CKPT_DIR / f"lspi_{feature_name}_pgn{pgn_label}_reg{REG}.npz"

        exp_id = f"lspi_{feature_name}_pgn{pgn_label}_seed{SEED}"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    py = sys.executable

    if args.force:
        remove_file(pgn_src)
        if anchor_src is not None:
            remove_file(anchor_src)
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
    print(f"PGN rows:          {pgn_samples}")
    print(f"Anchor rows:       {anchor_samples}")
    print(f"Source PGN rows:   {source_pgn_samples}")
    print(f"PGN source:        {pgn_src}")
    print(f"Anchor source:     {anchor_src}")
    print(f"Training dataset:  {train_src}")
    print(f"Shard dir:         {shard_dir}")
    print(f"Checkpoint:        {ckpt}")

    # 1. Build PGN source dataset
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
                str(MIN_ELO),
                "--max-samples",
                str(source_pgn_samples),
            ],
            env=env,
        )

    # 2. Build material anchors
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

        # 3. Mix dataset
        if train_src.exists() and not args.force:
            skip_step("mix dataset", train_src)
        else:
            run_step(
                "mix dataset",
                [
                    py,
                    "-m",
                    "scripts.mix_datasets",
                    "--src",
                    f"{pgn_src}:{pgn_samples}",
                    "--src",
                    f"{anchor_src}:{anchor_samples}",
                    "--out",
                    str(train_src),
                    "--seed",
                    str(SEED),
                ],
                env=env,
            )

    # 4. Split dataset
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

    # 5. Train
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

    if not args.skip_eval:
        # 6. Inspect checkpoint
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

        # 7. Position suite
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

        # 8. Random eval
        run_step(
            "self-play eval vs random",
            [
                py,
                "-m",
                "scripts.eval_selfplay",
                "--white",
                "lspi_v1",
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

        # 9. Material eval
        run_step(
            "self-play eval vs material",
            [
                py,
                "-m",
                "scripts.eval_selfplay",
                "--white",
                "lspi_v1",
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

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()