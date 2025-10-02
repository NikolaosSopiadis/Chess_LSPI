

class Board:
    
    def __init__(self, height=8, width=8):
        _height:    int = height
        _width:     int = width
        _grid_size: int = _height * _width

        _grid: list[int] = None
    