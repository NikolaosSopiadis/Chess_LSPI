

class Controller:
    
    STOPPED: int = 0
    RUNNING: int = 1
    
    def __init__(self) -> None:
        self._state: int = self.RUNNING
        
    def get_state(self) -> int:
        return self._state
    
    def update_state(self, state: int) -> None:
        self._state = state