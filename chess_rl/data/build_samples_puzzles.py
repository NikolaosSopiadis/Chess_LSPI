from __future__ import annotations
from typing import Optional, Set
import json
import gzip
from chess_core.board import Board
from chess_rl.uci import uci_to_move
from chess_rl.data.lichess_puzzles import iter_lichess_puzzles_csv
from chess_rl.rewards.v1_terminal_plus_potential import RewardSpec, step_reward
from chess_rl.features.base import FeatureExtractor

def build_samples_from_puzzles(
    puzzles_csv: str,
    out_jsonl_gz: str,
    feats: FeatureExtractor,
    *,
    include_themes: Optional[Set[str]] = None,
    min_rating: int = 1800,
    max_rows: int = 50_000,
) -> None:
    b = Board()

    with gzip.open(out_jsonl_gz, "wt", encoding="utf-8") as f:
        for row in iter_lichess_puzzles_csv(
            puzzles_csv,
            include_themes=include_themes,
            min_rating=min_rating,
            max_rows=max_rows,
        ):
            b.init_board(row.fen)

            # Walk through the provided PV moves; each ply gives one sample.
            for uci in row.moves_uci:
                done0, _ = b.game_end_state()
                if done0:
                    break

                move = uci_to_move(b, uci)

                # compute phi(s,a) via afterstate features:
                phi = feats.phi_sa(b, move)

                # advance to next
                before_fen = b.to_fen()
                u = b._do_move(move)
                try:
                    done, _ = b.game_end_state()
                    b_before = Board()
                    b_before.init_board(before_fen)
                    r = step_reward(b_before, b)
                    fen_next = b.to_fen()
                finally:
                    # keep the episode moving forward: do NOT undo here
                    pass

                rec = {
                    "phi": phi.tolist(),
                    "r": r,
                    "fen_next": fen_next,
                    "done": done,

                    # versions / provenance
                    "feature_version": feats.spec.version,
                    "reward_version": RewardSpec().version,
                    "source": "lichess_puzzles",
                    "puzzle_id": row.puzzle_id,
                    "themes": sorted(row.themes),
                    "rating": row.rating,
                }
                f.write(json.dumps(rec) + "\n")
