import pytest
from chess_core.board import Board

@pytest.mark.parametrize(
    "fen,expected",
    [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", True),   # K vs K
        ("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", True),  # K+N vs K
        ("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1", True), # K+B vs K
        ("4k3/8/8/8/8/8/8/2NNK3 w - - 0 1", True), # K+NN vs K

        # K+B vs K+B same color squares (c1 and f4 are same color)
        ("4k3/8/8/8/5B2/8/8/2B1K3 w - - 0 1", True),

        # Opposite-colored bishops (c1 and f5 are opposite) -> not dead
        ("4k3/8/8/5B2/8/8/8/2B1K3 w - - 0 1", False),

        # K+B+N vs K -> mate possible -> not dead
        ("4k3/8/8/8/8/8/8/1NB1K3 w - - 0 1", False),
    ],
)
def test_is_insufficient_material(fen: str, expected: bool):
    b = Board()
    b.init_board(fen)
    assert b.is_insufficient_material() is expected

def test_game_end_state_insufficient_material():
    b = Board()
    b.init_board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert b.game_end_state() == (True, "insufficient material")
