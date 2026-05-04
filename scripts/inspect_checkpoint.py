import numpy as np

ckpt = np.load("data/processed/checkpoints/lspi_v2_basic_pgn_200k_reg1e-1.iter09.npz", allow_pickle=True)
print(ckpt.files)

w = ckpt["w"]
names = [
    "bias",
    "material_total",
    "pawn_diff",
    "knight_diff",
    "bishop_diff",
    "rook_diff",
    "queen_diff",
    "white_K_castle",
    "white_Q_castle",
    "black_K_castle",
    "black_Q_castle",
    "side_to_move",
    "white_in_check",
    "black_in_check",
    "unused_14",
    "unused_15",
]

for i, (name, val) in enumerate(zip(names, w)):
    print(f"{i:2d} {name:18s} {val:+.6f}")