from __future__ import annotations

import numpy as np

from chess_core.board import Board
from chess_core.move import F_CASTLE
from chess_rl.lspi.lspi import iter_samples_jsonl_gz, _all_pseudo_moves, _castle_ok_pre
from chess_rl.features.v1_basic import V1BasicFeatures


def _sig(m) -> tuple[int, int, int, int]:
    return (m.src_square, m.dst_square, m.flags, m.promotion)


def legal_moves_fast(b: Board):
    stm_white = b._is_white_to_move
    pseudo = _all_pseudo_moves(b)

    legal = []
    for m in pseudo:
        if (m.flags & F_CASTLE) and (not _castle_ok_pre(b, stm_white, m)):
            continue

        u = b._do_move(m)
        try:
            king_sq = b._white_king_sq if stm_white else b._black_king_sq
            if king_sq == -1:
                continue
            if b.is_square_attacked(king_sq, by_white=not stm_white):
                continue
            legal.append(m)
        finally:
            b._undo_move(u)

    return legal


def check_fen(fen: str, *, verbose: bool = False) -> bool:
    b = Board()
    b.init_board(fen)

    ref = set(map(_sig, b.get_all_legal_moves()))
    test = set(map(_sig, legal_moves_fast(b)))

    if ref != test:
        if verbose:
            missing = ref - test
            extra = test - ref
            print("\nFEN mismatch:")
            print(fen)
            print(f"Missing ({len(missing)}): {sorted(missing)[:20]}")
            print(f"Extra   ({len(extra)}): {sorted(extra)[:20]}")
        return False

    return True


def check_dataset(samples_path: str, max_fens: int = 2000) -> None:
    bad = 0
    seen = 0
    for rec in iter_samples_jsonl_gz(samples_path):
        fen = str(rec["fen_next"])
        seen += 1
        ok = check_fen(fen, verbose=True)
        if not ok:
            bad += 1
            # stop early on first failure for debugging
            break
        if seen >= max_fens:
            break

    print(f"\nChecked {seen} FENs, failures={bad}")


def check_castling_cases() -> None:
    # 1) Castling should be legal (empty board with rooks/kings)
    fen_ok = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

    # 2) Through-check: black rook attacks f1 => white O-O should be illegal
    fen_through = "r3kr1r/8/8/8/8/8/8/R3K2R w KQ - 0 1"

    # 3) Out-of-check: white king in check by rook on e8 => castling illegal
    fen_out = "k3r3/8/8/8/8/8/8/R3K2R w KQ - 0 1"

    for name, fen in [("castle_ok", fen_ok), ("castle_through", fen_through), ("castle_out", fen_out)]:
        b = Board(); b.init_board(fen)
        ref = set(map(_sig, b.get_all_legal_moves()))
        test = set(map(_sig, legal_moves_fast(b)))
        print(f"\n{name}: match={ref == test}")
        if ref != test:
            print("Missing:", ref - test)
            print("Extra:", test - ref)

        # show whether any castle moves exist according to each
        ref_castle = [m for m in b.get_all_legal_moves() if (m.flags & F_CASTLE)]
        test_castle = [m for m in legal_moves_fast(b) if (m.flags & F_CASTLE)]
        print(f"ref castles={len(ref_castle)}  test castles={len(test_castle)}")


def check_phi_consistency(fen: str, n: int = 20) -> None:
    b = Board(); b.init_board(fen)
    feats = V1BasicFeatures()
    moves = b.get_all_legal_moves()

    for m in moves[:n]:
        pre_z = b._zkey
        phi1 = feats.phi_sa(b, m).copy()

        u = b._do_move(m)
        try:
            phi2 = feats.phi_sa_after_move(pre_z, m, b).copy()
        finally:
            b._undo_move(u)

        if not np.allclose(phi1, phi2):
            print("\nPHI mismatch!")
            print("FEN:", fen)
            print("Move sig:", _sig(m))
            print("phi1:", phi1)
            print("phi2:", phi2)
            return

    print("phi consistency OK")


if __name__ == "__main__":
    # edit these to your paths
    samples = "data/processed/samples/puzzles_v1_basic_r1800_50k.jsonl.gz"
    check_dataset(samples, max_fens=2000)
    check_castling_cases()

    # optional: pick one fen from failures or from dataset
    # check_phi_consistency("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
