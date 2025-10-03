# run_me.py
import pygame as pg
import pygame_gui

# ---------- layout ----------
BOARD_SIZE = 1200                      # left panel (square board)
SIDEBAR_W  = 500                      # right panel for UI
WIN_W, WIN_H = BOARD_SIZE + SIDEBAR_W, BOARD_SIZE

LIGHT = (240, 217, 181)
DARK  = (181, 136,  99)

def draw_board(surface: pg.Surface, ranks: int = 8, files: int = 8) -> None:
    sq = BOARD_SIZE // ranks
    for r in range(ranks):
        for f in range(files):
            color = LIGHT if (r + f) % 2 == 0 else DARK
            rect = pg.Rect(f * sq, r * sq, sq, sq)
            pg.draw.rect(surface, color, rect)

def main():
    pg.init()
    screen = pg.display.set_mode((WIN_W, WIN_H))
    pg.display.set_caption("Chess + pygame_gui sidebar")

    clock = pg.time.Clock()
    manager = pygame_gui.UIManager((WIN_W, WIN_H))

    # ----- sidebar root rect (we'll place all UI inside this area) -----
    sx = BOARD_SIZE  # sidebar x-start
    sidebar_rect = pg.Rect(sx, 0, SIDEBAR_W, WIN_H)

    # Container so we can position children relative to the sidebar
    sidebar = pygame_gui.elements.UIPanel(
        relative_rect=sidebar_rect,
        starting_height=1,
        manager=manager
    )

    # ----- widgets: title & player pickers -----
    title = pygame_gui.elements.UILabel(
        relative_rect=pg.Rect(16, 10, SIDEBAR_W - 32, 28),
        text="Game Controls",
        manager=manager,
        container=sidebar
    )

    white_label = pygame_gui.elements.UILabel(
        relative_rect=pg.Rect(16, 48, 110, 24),
        text="White:",
        manager=manager,
        container=sidebar
    )
    white_picker = pygame_gui.elements.UIDropDownMenu(
        options_list=['Human', 'Agent', 'Stockfish'],
        starting_option='Human',
        relative_rect=pg.Rect(120, 48, SIDEBAR_W - 136, 28),
        manager=manager,
        container=sidebar
    )

    black_label = pygame_gui.elements.UILabel(
        relative_rect=pg.Rect(16, 84, 110, 24),
        text="Black:",
        manager=manager,
        container=sidebar
    )
    black_picker = pygame_gui.elements.UIDropDownMenu(
        options_list=['Human', 'Agent', 'Stockfish'],
        starting_option='Human',
        relative_rect=pg.Rect(120, 84, SIDEBAR_W - 136, 28),
        manager=manager,
        container=sidebar
    )

    # ----- buttons -----
    new_btn = pygame_gui.elements.UIButton(
        relative_rect=pg.Rect(16, 128, SIDEBAR_W - 32, 34),
        text="New Game",
        manager=manager,
        container=sidebar
    )
    undo_btn = pygame_gui.elements.UIButton(
        relative_rect=pg.Rect(16, 170, SIDEBAR_W - 32, 34),
        text="Undo (U)",
        manager=manager,
        container=sidebar
    )
    settings_btn = pygame_gui.elements.UIButton(
        relative_rect=pg.Rect(16, 212, SIDEBAR_W - 32, 34),
        text="Settings…",
        manager=manager,
        container=sidebar
    )

    # ----- scrollable move list -----
    moves_label = pygame_gui.elements.UILabel(
        relative_rect=pg.Rect(16, 260, SIDEBAR_W - 32, 24),
        text="Moves",
        manager=manager,
        container=sidebar
    )

    moves_container = pygame_gui.elements.UIScrollingContainer(
        relative_rect=pg.Rect(16, 288, SIDEBAR_W - 32, WIN_H - 304),
        manager=manager,
        container=sidebar
    )

    # A tall text box inside the scrolling container (we’ll update html_text)
    moves_text = pygame_gui.elements.UITextBox(
        html_text="",
        relative_rect=pg.Rect(0, 0, SIDEBAR_W - 16, 1200),
        container=moves_container,
        manager=manager,
        object_id="#moves_text"
    )

    # demo move buffer
    san_moves: list[str] = []

    # settings modal (created on demand)
    settings_window = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                continue

            # keyboard shortcut example
            if event.type == pg.KEYDOWN and event.key == pg.K_u:
                # simulate undo by popping a move
                if san_moves:
                    san_moves.pop()
                    moves_text.html_text = "<br>".join(san_moves)
                    moves_text.rebuild()

            # Hand off to pygame_gui
            manager.process_events(event)

            # Handle pygame_gui user events
            if event.type == pg.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == new_btn:
                        san_moves.clear()
                        moves_text.html_text = ""
                        moves_text.rebuild()
                    elif event.ui_element == undo_btn:
                        if san_moves:
                            san_moves.pop()
                            moves_text.html_text = "<br>".join(san_moves)
                            moves_text.rebuild()
                    elif event.ui_element == settings_btn:
                        if settings_window is None or settings_window.alive is False:
                            settings_window = pygame_gui.elements.UIWindow(
                                rect=pg.Rect((WIN_W // 2 - 180, WIN_H // 2 - 120), (360, 240)),
                                manager=manager,
                                window_display_title="Settings",
                                object_id="#settings_window"
                            )
                            # A couple of example controls inside settings
                            pygame_gui.elements.UILabel(
                                relative_rect=pg.Rect(16, 16, 120, 24),
                                text="Theme:",
                                container=settings_window,
                                manager=manager
                            )
                            pygame_gui.elements.UIDropDownMenu(
                                options_list=["Classic", "Blue", "High Contrast"],
                                starting_option="Classic",
                                relative_rect=pg.Rect(120, 16, 200, 28),
                                container=settings_window,
                                manager=manager
                            )
                            pygame_gui.elements.UIButton(
                                relative_rect=pg.Rect(120, 160, 100, 32),
                                text="OK",
                                container=settings_window,
                                manager=manager
                            )

                elif event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == white_picker:
                        # hook into your controller: set white player type
                        pass
                    elif event.ui_element == black_picker:
                        # set black player type
                        pass

        # Demo: append a fake move every 2 seconds so you can see scrolling
        if pg.time.get_ticks() % 2000 < 16:
            san_moves.append(f"{len(san_moves)+1}. e2e4")
            moves_text.html_text = "<br>".join(san_moves)
            moves_text.rebuild()

        # ----- update & draw -----
        manager.update(dt)

        # draw board on the left
        screen.fill((30, 30, 30))
        draw_board(screen, 8, 8)

        # draw UI on top (right)
        manager.draw_ui(screen)

        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    main()
