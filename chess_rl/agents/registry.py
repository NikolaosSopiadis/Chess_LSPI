from __future__ import annotations

from pathlib import Path

from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent


HUMAN = "Human"
RANDOM = "Random legal"
LSPI_V1 = "LSPI v1"

# Adjust this path whenever you want to test a different checkpoint.
LSPI_V1_CHECKPOINT = Path("data/processed/checkpoints/lspi_v1.9_profiling.npz")


def player_options() -> list[str]:
    return [
        HUMAN,
        RANDOM,
        LSPI_V1,
    ]


def make_player(player_id: str) -> Agent | None:
    """
    Returns:
        None for human players.
        Agent instance for AI players.
    """
    if player_id == HUMAN:
        return None

    if player_id == RANDOM:
        return RandomAgent()

    if player_id == LSPI_V1:
        if not LSPI_V1_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v1 checkpoint: {LSPI_V1_CHECKPOINT}")
        return LSPIV1Agent.load(str(LSPI_V1_CHECKPOINT))

    raise ValueError(f"Unknown player: {player_id!r}")