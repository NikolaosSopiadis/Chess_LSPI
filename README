# Chess Q-Learning / LSPI Agent

A custom chess engine and reinforcement-learning project focused on training chess agents with **Least-Squares Policy Iteration (LSPI)**.

The project contains:

* a custom chess core implementation,
* a lightweight graphical chess interface,
* feature extractors for LSPI afterstate evaluation,
* dataset generation scripts,
* LSPI training scripts,
* trained model checkpoints,
* model evaluation and comparison tools,
* search-based LSPI agents.

The main goal of the project is to study how different chess feature representations affect the playing strength of an LSPI-based chess agent.

---

## Project overview

The project is built around three main components:

```text
chess_core/
    Custom chess engine:
    board representation, legal moves, move execution/undo,
    FEN handling, check/checkmate detection, repetition tracking.

gui/
    Lightweight graphical interface:
    human play, agent play, move history, board visualization.

chess_rl/
    Reinforcement-learning components:
    LSPI agents, feature extractors, search agents, training utilities.
```

The reinforcement learning pipeline uses **afterstate evaluation**. For every legal move, the engine temporarily applies the move, extracts features from the resulting position, and evaluates that afterstate using a learned linear model:

```text
score(move) = wᵀ φ(afterstate)
```

The final agents combine the learned LSPI evaluator with tactical safety checks and depth-limited search.

---

## Features

### Chess engine

The custom chess core supports:

* legal move generation,
* check and checkmate detection,
* castling,
* en-passant,
* promotion,
* stalemate,
* fifty-move rule,
* threefold repetition,
* FEN import/export,
* reversible move execution,
* Zobrist hashing,
* repetition tracking,
* helper APIs for tactical feature extraction.

The engine is designed to be used both interactively through the GUI and programmatically by the training and evaluation scripts.

---

### Reinforcement learning

The LSPI pipeline supports:

* offline dataset generation,
* PGN-based training samples,
* synthetic material anchor samples,
* opening/center anchor samples,
* mixed datasets,
* streaming JSONL-GZ training data,
* regularized LSPI matrix solving,
* checkpoint saving/loading,
* feature-version tracking,
* model inspection,
* self-play evaluation,
* model-vs-model matrix evaluation.

---

### Agents

The project includes several types of agents:

* random legal-move agent,
* material greedy agent,
* material minimax agent,
* plain LSPI agent,
* tactical LSPI agent,
* search-based LSPI agent.

The strongest models use LSPI as an evaluation function inside a search procedure.

---

## Environment setup

The project is written in Python. It is recommended to run it inside a virtual environment so that the project dependencies do not interfere with the rest of the system.

From the repository root:

```bash
cd Chess_LSPI
```

Create a virtual environment:

```bash
python -m venv .venv
```

On some systems, the command may be:

```bash
python3 -m venv .venv
```

Activate the virtual environment.

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When the environment is active, the terminal usually shows `(.venv)` at the beginning of the prompt.

To deactivate the environment later:

```bash
deactivate
```

---

## Running the GUI

The graphical interface can be used to manually play chess, test the custom engine, or observe games between agents.

Run the GUI from the repository root using the project’s GUI entry point. For example:

```bash
python -m gui.main
```

If the project uses a different top-level launcher, use:

```bash
python main.py
```

The GUI is mainly intended for:

* manual testing,
* debugging move generation,
* visualizing agent behavior,
* playing against trained agents,
* watching agent-vs-agent games.

The reinforcement learning pipeline itself does not require the GUI.

---

## Training a new LSPI model

Training a model usually consists of:

```text
1. Generate LSPI training samples.
2. Mix the generated datasets.
3. Train LSPI.
4. Save a checkpoint.
5. Inspect and evaluate the model.
```

The easiest way to run a complete experiment is with:

```bash
python -m scripts.run_lspi_experiment \
  --agent v10 \
  --pgn 150000 \
  --anchor 50000 \
  --min-elo 2000
```

This high-level script automates the common workflow:

* generate PGN samples,
* generate anchor samples,
* mix datasets,
* train the LSPI model,
* save a checkpoint,
* optionally evaluate the model.

A larger example:

```bash
python -m scripts.run_lspi_experiment \
  --agent v10 \
  --pgn 150000 \
  --anchor 50000 \
  --min-elo 2000 \
  --matrix-label v10_new \
  --matrix-kind tactical \
  --matrix-games 80 \
  --matrix-workers 12 \
  --matrix-chunk-size 2
```

Trained checkpoints are saved under:

```text
data/processed/checkpoints/
```

---

## Manual training pipeline

The pipeline can also be run manually for more control.

A typical manual workflow is:

```bash
python -m scripts.build_dataset_pgn ...
python -m scripts.build_dataset_material_anchor ...
python -m scripts.mix_datasets ...
python -m scripts.train_lspi ...
python -m scripts.inspect_checkpoint ...
python -m scripts.eval_model_matrix ...
```

This is useful when experimenting with:

* different feature versions,
* different PGN sample counts,
* different anchor ratios,
* different minimum Elo filters,
* different reward versions,
* different evaluation settings.

---

## Evaluating models

The main model comparison script is:

```bash
python -m scripts.eval_model_matrix
```

It can compare many models either as a full round-robin or as an adjacent ladder.

Example adjacent ladder evaluation:

```bash
python -m scripts.eval_model_matrix \
  --pair-mode adjacent \
  --compact-report \
  --compact-json \
  --model v1:plain:data/processed/checkpoints/lspi_v1_basic_pgn_200k_reg1e-1.npz \
  --model v2:plain:data/processed/checkpoints/lspi_v2_1_basic_mix_pgn900k_anchor100k_reg1e-1.npz \
  --model v3_2_search:search:data/processed/checkpoints/lspi_v3_basic_mix_pgn750k_anchor250k_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v4_2_search:search:data/processed/checkpoints/lspi_v4_slim_mix_pgn150k_anchor50k_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v5_2_center_1M:search:data/processed/checkpoints/lspi_v5_center_mix_pgn650k_anchor250k_center100k_2000elo_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v6_2_attackmap:search:data/processed/checkpoints/lspi_v6_attackmap_mix_pgn150k_anchor50k_2000elo_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v7_2_api_1M:search:data/processed/checkpoints/lspi_v7_api_tactics_mix_pgn750k_anchor250k_2000elo_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v8_2_clean:search:data/processed/checkpoints/lspi_v8_api_tactics_clean_mix_pgn150k_anchor50k_2000elo_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v9_2_response:search:data/processed/checkpoints/lspi_v9_response_tactics_mix_pgn150k_anchor50k_2000elo_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --model v10_2_fast:search:data/processed/checkpoints/lspi_v10_response_fast_mix_pgn150k_anchor50k_2000elo_reg1e-1.npz:depth=2,max_branch=none,draw=1,tactical=1 \
  --games 80 \
  --position-source suite \
  --max-plies 250 \
  --random-openings 6 \
  --seed 10 \
  --workers 12 \
  --chunk-size 2 \
  --json-out data/processed/eval/final_ladder_search_seed10.json
```

The evaluation output includes:

* win/draw/loss results,
* score matrix,
* W-D-L matrix,
* average game length,
* termination reasons,
* optional opening/style statistics,
* optional JSON export.

---

## Main scripts

### Dataset generation

| Script                                   | Purpose                                                       |
| ---------------------------------------- | ------------------------------------------------------------- |
| `build_dataset_pgn.py`                   | Converts PGN games into LSPI training samples.                |
| `build_dataset_material_anchor.py`       | Generates synthetic material-balance anchor samples.          |
| `build_dataset_opening_center_anchor.py` | Generates opening and center-control anchor samples.          |
| `mix_datasets.py`                        | Combines multiple datasets into a single mixed training file. |
| `split_dataset.py`                       | Splits datasets into smaller subsets.                         |

### Training

| Script                   | Purpose                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `train_lspi.py`          | Main LSPI training script. Accumulates the LSPI matrices, solves for the weight vector, and saves checkpoints. |
| `run_lspi_experiment.py` | High-level experiment runner for dataset generation, mixing, training, and evaluation.                         |
| `inspect_checkpoint.py`  | Prints learned feature weights, material ratios, and checkpoint metadata.                                      |

### Evaluation

| Script                 | Purpose                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `eval_selfplay.py`     | Plays games between agents and records results.                         |
| `eval_model_matrix.py` | Compares several models in a round-robin or adjacent-ladder evaluation. |

### Support scripts

| Script type      | Purpose                                                              |
| ---------------- | -------------------------------------------------------------------- |
| Ablation scripts | Test the effect of removing feature groups or modifying checkpoints. |
| Utility scripts  | Help inspect datasets, checkpoints, or experiment outputs.           |

---

## Feature versions

The project evolved through several feature versions:

| Version                | Main idea                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `v1_basic`             | First basic LSPI feature set.                                                       |
| `v2_1_basic`           | Improved basic/material feature set.                                                |
| `v3_basic`             | Stronger baseline feature set, later used with search.                              |
| `v4_slim`              | Compact stable baseline with material, draw, development, and king-safety features. |
| `v5_center`            | Adds center-control and opening-center features.                                    |
| `v6_attackmap`         | Adds attack-map and pressure features.                                              |
| `v7_api_tactics`       | Adds legal-move tactical features.                                                  |
| `v8_api_tactics_clean` | Removes the full attack-map block and keeps selected tactical features.             |
| `v9_response_tactics`  | Adds concrete side-specific and response tactical features.                         |
| `v10_response_fast`    | Pruned faster version of the response-tactical feature set.                         |

The final experiments showed that adding search produced the largest strength improvement, while the strongest late-stage feature improvement came from the response-oriented tactical features in `v9`.

---

## Final model comparison

The final adjacent-ladder evaluation used 160 games per matchup and produced the following main conclusions:

* `v3.2 Search` strongly outperformed the earlier non-search model.
* `v4.2 Search` improved over the previous searched baseline.
* `v5.2 Center` did not improve over `v4.2`.
* `v6.2 Attackmap` improved over the center-feature model.
* `v7.2 API Tactics` improved over the attack-map model.
* `v8.2 Clean` was simpler but weaker than `v7.2`.
* `v9.2 Response` produced the strongest late-stage feature improvement.
* `v10.2 Fast` traded some strength for a smaller and faster feature set.

In the final ladder, the strongest model was `v9_2_response`, while `v10_2_fast` represents a faster deployment-oriented version.

---

## Repository data

Large generated files such as datasets, checkpoints, and evaluation results are expected under:

```text
data/processed/
```

Typical subdirectories are:

```text
data/processed/datasets/
data/processed/checkpoints/
data/processed/eval/
```

Depending on repository size constraints, large generated files may be excluded from version control and regenerated using the scripts.
