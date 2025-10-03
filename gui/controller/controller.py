from gui.view.main_window import MainWindow
    
class Controller:
    
    STOPPED: int = 0
    RUNNING: int = 1
    
    def __init__(self) -> None:
        self._state: int = self.RUNNING
        self._ranks: int = 8
        self._files: int = 8
        
        self._view: MainWindow = MainWindow(self, 800, 600, "Chess")
        
    def get_state(self) -> int:
        return self._state
    
    def update_state(self, state: int) -> None:
        self._state = state
        
    def get_ranks(self) -> int:
        return self._ranks
    
    def get_files(self) -> int:
        return self._files
    
    # TODO: Remove this and interact directly with controller
    def get_view(self) -> MainWindow:
        return self._view