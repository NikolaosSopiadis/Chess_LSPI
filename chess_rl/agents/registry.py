from __future__ import annotations

from pathlib import Path

from chess_rl.agents.base import Agent
from chess_rl.agents.random import RandomAgent
from chess_rl.agents.lspi_v1 import LSPIV1Agent
from chess_rl.agents.material_greedy import MaterialGreedyAgent
from chess_rl.agents.lspi_tactical import LSPITacticalAgent
from chess_rl.agents.lspi_search import LSPISearchAgent


HUMAN = "Human"
LSPI_V6_SEARCH_100k = "LSPI v6.2 100k Search"
LSPI_V5_SEARCH_1M = "LSPI v5.2 1M Search" # center
LSPI_V5_SEARCH_100k = "LSPI v5.2 100k Search" #center
LSPI_V4_SEARCH_1M = "LSPI v4.2 Search"
LSPI_V4_TACTICAL_1M = "LSPI v4.1 Tactical"    
LSPI_V4_1M = "LSPI v4.0"    
LSPI_V3_SEARCH_1M = "LSPI v3.2 Search"
LSPI_V3_TACTICAL_1M = "LSPI v3.1 Tactical"    
LSPI_V3_1M = "LSPI v3.0"    
LSPI_V1 = "LSPI v1"
GREEDY = "Material greedy"
RANDOM = "Random legal"

# Adjust this path whenever you want to test a different checkpoint.
LSPI_V1_CHECKPOINT = Path("data/processed/checkpoints/lspi_v1_basic_pgn_200k_reg1e-1.npz")
LSPI_V3_1M_CHECKPOINT = Path("data/processed/checkpoints/lspi_v3_basic_mix_pgn750k_anchor250k_reg1e-1.npz")
LSPI_V4_1M_CHECKPOINT = Path("data/processed/checkpoints/lspi_v4_slim_mix_pgn150k_anchor50k_reg1e-1.npz")
LSPI_V5_100k_CHECKPOINT = Path("data/processed/checkpoints/lspi_v5_center_mix_pgn135k_material50k_center15k_2000elo_reg1e-1.npz")
LSPI_V5_1M_CHECKPOINT = Path("data/processed/checkpoints/lspi_v5_center_mix_pgn650k_anchor250k_center100k_2000elo_reg1e-1.npz")
LSPI_V5_100k_CHECKPOINT = Path("data/processed/checkpoints/lspi_v5_center_mix_pgn135k_material50k_center15k_2000elo_reg1e-1.npz")
LSPI_V6_100k_CHECKPOINT = Path("data/processed/checkpoints/lspi_v6_attackmap_mix_pgn150k_anchor50k_2000elo_reg1e-1.npz")

def player_options() -> list[str]:
    return [
        HUMAN,
        LSPI_V6_SEARCH_100k,
        LSPI_V5_SEARCH_100k,
        LSPI_V5_SEARCH_1M,
        LSPI_V4_SEARCH_1M,
        LSPI_V4_TACTICAL_1M,
        LSPI_V4_1M,
        LSPI_V3_SEARCH_1M,
        LSPI_V3_TACTICAL_1M,
        LSPI_V3_1M,
        LSPI_V1,
        GREEDY,
        RANDOM,
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

    if player_id == GREEDY:
        return MaterialGreedyAgent()

    if player_id == LSPI_V1:
        if not LSPI_V1_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v1 checkpoint: {LSPI_V1_CHECKPOINT}")
        return LSPIV1Agent.load(str(LSPI_V1_CHECKPOINT))

    if player_id == LSPI_V3_1M:
        if not LSPI_V3_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v3 checkpoint: {LSPI_V3_1M_CHECKPOINT}")
        return LSPIV1Agent.load(str(LSPI_V3_1M_CHECKPOINT))

    if player_id == LSPI_V3_TACTICAL_1M:
        if not LSPI_V3_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v3 1M checkpoint: {LSPI_V3_1M_CHECKPOINT}")
        return LSPITacticalAgent.load(str(LSPI_V3_1M_CHECKPOINT))
    
    if player_id == LSPI_V3_SEARCH_1M:
        if not LSPI_V3_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v3 checkpoint: {LSPI_V3_1M_CHECKPOINT}")

        return LSPISearchAgent.load(
            str(LSPI_V3_1M_CHECKPOINT),
            depth=2,
            max_branch=None,
            use_draw_safety=True,
            use_tactical_safety=False,
        ) 

    if player_id == LSPI_V4_1M:
        if not LSPI_V4_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v4 1M checkpoint: {LSPI_V4_1M_CHECKPOINT}")
        return LSPIV1Agent.load(str(LSPI_V4_1M_CHECKPOINT))

    if player_id == LSPI_V4_TACTICAL_1M:
        if not LSPI_V4_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v4 1M checkpoint: {LSPI_V4_1M_CHECKPOINT}")
        return LSPITacticalAgent.load(str(LSPI_V4_1M_CHECKPOINT))
    
    if player_id == LSPI_V4_SEARCH_1M:
        if not LSPI_V4_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v4 1M checkpoint: {LSPI_V4_1M_CHECKPOINT}")

        return LSPISearchAgent.load(
            str(LSPI_V4_1M_CHECKPOINT),
            depth=2,
            max_branch=None,
            use_draw_safety=True,
            use_tactical_safety=False,
        )

    if player_id == LSPI_V5_SEARCH_100k:
        if not LSPI_V5_100k_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v5 100k checkpoint: {LSPI_V5_100k_CHECKPOINT}")

        return LSPISearchAgent.load(
            str(LSPI_V5_100k_CHECKPOINT),
            depth=2,
            max_branch=None,
            use_draw_safety=True,
            use_tactical_safety=False,
        )

    if player_id == LSPI_V5_SEARCH_1M:
        if not LSPI_V5_1M_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v5 1M checkpoint: {LSPI_V5_1M_CHECKPOINT}")

        return LSPISearchAgent.load(
            str(LSPI_V5_1M_CHECKPOINT),
            depth=2,
            max_branch=None,
            use_draw_safety=True,
            use_tactical_safety=False,
        )

    if player_id == LSPI_V6_SEARCH_100k:
        if not LSPI_V6_100k_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing LSPI v6 100k checkpoint: {LSPI_V6_100k_CHECKPOINT}")

        return LSPISearchAgent.load(
            str(LSPI_V6_100k_CHECKPOINT),
            depth=2,
            max_branch=None,
            use_draw_safety=True,
            use_tactical_safety=False,
        )

    raise ValueError(f"Unknown player: {player_id!r}")