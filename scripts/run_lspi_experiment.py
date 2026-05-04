from __future__ import annotations

import argparse
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


# -----------------------------
# Constants we currently reuse
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
LOG_DIR = Path("data/processed/logs")


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
      v3      -> v3_basic
      v2_1    -> v2_1_basic
      v3_basic -> v3_basic
    """
    name = name.strip()

    if name.endswith("_basic"):
        return name

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
        path.unlink()


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def remove_checkpoint_family(path: Path) -> None:
    remove_file(path)

    # lspi_xxx.npz -> lspi_xxx.iter*.npz
    pattern = f"{path.stem}.iter*.npz"
    for p in path.parent.glob(pattern):
        p.unlink()


def shards_exist(shard_dir: Path, expected: int) -> bool:
    if not shard_dir.exists():
        return False
    return len(list(shard_dir.glob("*.jsonl.gz"))) >= expected


def write_header(log_f, title: str) -> None:
    log_f.write("\n")
    log_f.write("=" * 100 + "\n")
    log_f.write(title + "\n")
    log_f.write("=" * 100 + "\n")
    log_f.flush()


def tail(path: Path, lines: int = 80) -> str:
    try:
        data = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(data[-lines:])


def run_logged(
    *,
    desc: str,
    cmd: list[str],
    log_f,
    log_path: Path,
    env: dict[str, str],
) -> None:
    print(f"[run]  {desc}")
    write_header(log_f, desc)
    log_f.write("$ " + shlex.join(cmd) + "\n\n")
    log_f.flush()

    proc = subprocess.run(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    log_f.flush()

    if proc.returncode != 0:
        print()
        print(f"[fail] {desc}")
        print(f"Log: {log_path}")
        print()
        print("Last log lines:")
        print(tail(log_path))
        raise SystemExit(proc.returncode)

    print(f"[ok]   {desc}")


def skip_logged(desc: str, log_f) -> None:
    print(f"[skip] {desc}")
    write_header(log_f, f"SKIP: {desc}")


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
        help="Regenerate data, shards, training, and overwrite existing log.",
    )

    args = ap.parse_args()

    feature_name = normalize_agent_name(args.agent)
    ds_label = dataset_feature_label(feature_name)

    pgn_samples = args.pgn
    anchor_samples = args.anchor
    total_source_pgn_samples = pgn_samples + anchor_samples

    pgn_label = sample_label(pgn_samples)
    anchor_label = sample_label(anchor_samples)
    source_label = sample_label(total_source_pgn_samples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Source PGN file used both for PGN rows and anchor construction.
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

    log_path = LOG_DIR / f"{exp_id}.log"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    py = sys.executable

    # Clean generated artifacts if requested.
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
    print(f"Experiment: {exp_id}")
    print(f"Feature:    {feature_name}")
    print(f"PGN rows:   {pgn_samples}")
    print(f"Anchors:    {anchor_samples}")
    print(f"Log:        {log_path}")
    print()

    with log_path.open("w", encoding="utf-8") as log_f:
        write_header(log_f, "EXPERIMENT CONFIG")
        log_f.write(f"experiment: {exp_id}\n")
        log_f.write(f"feature_name: {feature_name}\n")
        log_f.write(f"dataset_label: {ds_label}\n")
        log_f.write(f"pgn_samples: {pgn_samples}\n")
        log_f.write(f"anchor_samples: {anchor_samples}\n")
        log_f.write(f"source_pgn_samples: {total_source_pgn_samples}\n")
        log_f.write(f"pgn_src: {pgn_src}\n")
        log_f.write(f"anchor_src: {anchor_src}\n")
        log_f.write(f"train_src: {train_src}\n")
        log_f.write(f"shard_dir: {shard_dir}\n")
        log_f.write(f"checkpoint: {ckpt}\n")
        log_f.flush()

        # 1. Build PGN source dataset
        if pgn_src.exists() and not args.force:
            skip_logged(f"build PGN source dataset: {pgn_src}", log_f)
        else:
            run_logged(
                desc=f"build PGN source dataset: {pgn_src}",
                cmd=[
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
                    str(total_source_pgn_samples),
                ],
                log_f=log_f,
                log_path=log_path,
                env=env,
            )

        # 2. Build material anchors, if requested
        if anchor_samples > 0:
            assert anchor_src is not None

            if anchor_src.exists() and not args.force:
                skip_logged(f"build material anchors: {anchor_src}", log_f)
            else:
                run_logged(
                    desc=f"build material anchors: {anchor_src}",
                    cmd=[
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
                    log_f=log_f,
                    log_path=log_path,
                    env=env,
                )

            # 3. Mix dataset
            if train_src.exists() and not args.force:
                skip_logged(f"mix dataset: {train_src}", log_f)
            else:
                run_logged(
                    desc=f"mix dataset: {train_src}",
                    cmd=[
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
                    log_f=log_f,
                    log_path=log_path,
                    env=env,
                )

        # 4. Split dataset
        if shards_exist(shard_dir, SHARDS) and not args.force:
            skip_logged(f"split dataset: {shard_dir}", log_f)
        else:
            run_logged(
                desc=f"split dataset: {shard_dir}",
                cmd=[
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
                log_f=log_f,
                log_path=log_path,
                env=env,
            )

        # 5. Train
        if ckpt.exists() and not args.force:
            skip_logged(f"train LSPI: {ckpt}", log_f)
        else:
            run_logged(
                desc=f"train LSPI: {ckpt}",
                cmd=[
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
                log_f=log_f,
                log_path=log_path,
                env=env,
            )

        # 6. Inspect checkpoint
        run_logged(
            desc="inspect checkpoint",
            cmd=[
                py,
                "-m",
                "scripts.inspect_checkpoint",
                str(ckpt),
            ],
            log_f=log_f,
            log_path=log_path,
            env=env,
        )

        # 7. Position suite
        run_logged(
            desc="position suite",
            cmd=[
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
            log_f=log_f,
            log_path=log_path,
            env=env,
        )

        # 8. Random eval
        run_logged(
            desc="self-play eval vs random",
            cmd=[
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
            log_f=log_f,
            log_path=log_path,
            env=env,
        )

        # 9. Material eval
        run_logged(
            desc="self-play eval vs material",
            cmd=[
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
            log_f=log_f,
            log_path=log_path,
            env=env,
        )

    print()
    print("[done]")
    print(f"Checkpoint: {ckpt}")
    print(f"Log:        {log_path}")
    print()
    print("To paste the full output:")
    print(f"  cat {shlex.quote(str(log_path))}")


if __name__ == "__main__":
    main()