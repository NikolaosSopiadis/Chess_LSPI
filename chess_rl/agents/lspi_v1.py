from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from chess_core.board import Board
from chess_core.move import Move
from chess_rl.agents.base import Agent, AgentInfo
from chess_rl.features.registry import get as get_features
from chess_rl.policy.greedy import greedy_move
from chess_rl.rewards.v1_terminal_plus_potential import material_potential


@dataclass
class LSPIV1Agent(Agent):
    info: AgentInfo
    w: np.ndarray
    feature_name: str = "v1_basic"  # registry key

    def pick_move(self, board: Board) -> Move:
        feats = get_features(self.feature_name)
        return greedy_move(board, self.w, feats)

    def save(self, path: str) -> None:
        path = str(path)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.info.name,
            "version": self.info.version,
            "feature_name": self.feature_name,
        }

        np.savez_compressed(path, w=self.w.astype(np.float64), meta=json.dumps(meta))

    @classmethod
    def load(cls, path: str) -> "LSPIV1Agent":
        data = np.load(path, allow_pickle=False)
        w = data["w"]
        meta = json.loads(str(data["meta"]))
        return cls(
            info=AgentInfo(name=meta["name"], version=meta["version"]),
            w=w,
            feature_name=meta.get("feature_name", "v1_basic"),
        )
