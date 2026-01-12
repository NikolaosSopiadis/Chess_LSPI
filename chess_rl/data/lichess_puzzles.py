# chess_rl/data/lichess_puzzles.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable, Iterator, Optional, Set

@dataclass(frozen=True)
class PuzzleRow:
    puzzle_id: str
    fen: str
    moves_uci: list[str]
    rating: int
    themes: set[str]

def iter_lichess_puzzles_csv(
    csv_path: str | Path,
    *,
    include_themes: Optional[Set[str]] = None,
    exclude_themes: Optional[Set[str]] = None,
    min_rating: Optional[int] = None,
    max_rows: Optional[int] = None,
    require_all: bool = False
) -> Iterator[PuzzleRow]:
    """
    Reads Lichess puzzles CSV:
    PuzzleId,FEN,Moves,Rating,...,Themes,...
    Themes are space-separated.
    """
    csv_path = Path(csv_path)
    n = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return

        # Indices per lichess format
        # 0 PuzzleId, 1 FEN, 2 Moves, 3 Rating, 7 Themes
        for row in reader:
            if not row:
                continue
            puzzle_id = row[0]
            fen = row[1]
            moves = row[2].split()
            rating = int(row[3])
            themes = set(row[7].split()) if len(row) > 7 and row[7] else set()

            if min_rating is not None and rating < min_rating:
                continue
            if include_themes is not None:
                if require_all and not include_themes.issubset(themes):
                    continue
                if not require_all and themes.isdisjoint(include_themes):
                    continue
            if exclude_themes is not None and not themes.isdisjoint(exclude_themes):
                continue

            yield PuzzleRow(puzzle_id, fen, moves, rating, themes)

            n += 1
            if max_rows is not None and n >= max_rows:
                return
