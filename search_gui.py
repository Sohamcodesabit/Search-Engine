import pygame
import sys
import threading
from main import search, documents

# --- Helper to capture search results ---
class SearchResultCatcher:
    def __init__(self):
        self.results = []

    def capture(self, query, top_n=5):
        self.results = []
        def print_override(*args, **kwargs):
            self.results.append(' '.join(str(a) for a in args))
        # Patch print temporarily
        import builtins
        orig_print = builtins.print
        builtins.print = print_override
        try:
            search(query, top_n)
        finally:
            builtins.print = orig_print

    def get_results(self):
        return self.results

# --- Pygame GUI ---
pygame.init()
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h
FONT = pygame.font.SysFont("arial", 36)
SMALL_FONT = pygame.font.SysFont("arial", 24)
TITLE_FONT = pygame.font.SysFont("arial", 32, bold=True)
BG_COLOR = (176, 208, 176)  # Sage green
TEXT_COLOR = (30, 30, 40)
INPUT_BG = (255, 192, 203)  # Pink
BUTTON_BG = (255, 192, 203)  # Pink
BUTTON_HOVER = (235, 172, 183)  # Slightly darker pink
RESULT_BG = (255, 192, 203)  # Pink
RESULT_BOX_BG = (255, 192, 203)  # Pink
BORDER_COLOR = (30, 30, 40)
TITLE_BAR_BG = (255, 192, 203)  # Pink
TITLE_BAR_HEIGHT = 48
TITLE_BAR_BTN_BG = (200, 60, 60)
TITLE_BAR_BTN_HOVER = (255, 80, 80)
TITLE_BAR_BTN_MIN_BG = (60, 120, 200)
TITLE_BAR_BTN_MIN_HOVER = (90, 150, 240)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Document Search")

# Layout positions and sizes
margin = 60
input_box = pygame.Rect(margin, TITLE_BAR_HEIGHT + 52, WIDTH - 2*margin - 200, 50)
button_box = pygame.Rect(WIDTH - margin - 180, TITLE_BAR_HEIGHT + 52, 180, 50)
results_label_box = pygame.Rect(margin, TITLE_BAR_HEIGHT + 130, WIDTH - 2*margin, 50)
results_content_box = pygame.Rect(margin, TITLE_BAR_HEIGHT + 200, WIDTH - 2*margin, HEIGHT - (TITLE_BAR_HEIGHT + 200) - margin)
scroll_offset = 0

# Title bar buttons
close_btn = pygame.Rect(WIDTH - 60, 10, 40, 28)
min_btn = pygame.Rect(WIDTH - 110, 10, 40, 28)

user_text = ""
active = False
results = []
catcher = SearchResultCatcher()
last_query = ""
cursor_visible = True
cursor_timer = 0

def wrap_text(text, font, max_width):
    """
    Splits text into a list of lines so that each line fits within max_width pixels.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def draw_title_bar():
    # Draw title bar background
    pygame.draw.rect(screen, TITLE_BAR_BG, (0, 0, WIDTH, TITLE_BAR_HEIGHT))
    # Draw title text
    title = TITLE_FONT.render("Document Search", True, TEXT_COLOR)
    title_rect = title.get_rect(midleft=(margin, TITLE_BAR_HEIGHT//2))
    screen.blit(title, title_rect)
    # Draw Minimize button
    mouse_pos = pygame.mouse.get_pos()
    min_color = TITLE_BAR_BTN_MIN_HOVER if min_btn.collidepoint(mouse_pos) else TITLE_BAR_BTN_MIN_BG
    pygame.draw.rect(screen, min_color, min_btn, border_radius=8)
    pygame.draw.rect(screen, BORDER_COLOR, min_btn, 2, border_radius=8)
    min_text = TITLE_FONT.render("_", True, (255,255,255))
    min_rect = min_text.get_rect(center=min_btn.center)
    screen.blit(min_text, min_rect)
    # Draw Close button
    close_color = TITLE_BAR_BTN_HOVER if close_btn.collidepoint(mouse_pos) else TITLE_BAR_BTN_BG
    pygame.draw.rect(screen, close_color, close_btn, border_radius=8)
    pygame.draw.rect(screen, BORDER_COLOR, close_btn, 2, border_radius=8)
    close_text = TITLE_FONT.render("×", True, (255,255,255))
    close_rect = close_text.get_rect(center=close_btn.center)
    screen.blit(close_text, close_rect)

def draw():
    screen.fill(BG_COLOR)
    draw_title_bar()
    # Outer border
    pygame.draw.rect(screen, BORDER_COLOR, (10, 10, WIDTH-20, HEIGHT-20), 2, border_radius=30)
    # Input box
    pygame.draw.rect(screen, INPUT_BG, input_box, border_radius=12)
    pygame.draw.rect(screen, BORDER_COLOR, input_box, 2, border_radius=12)
    # Vertically center text in input box
    txt_surface = FONT.render(user_text, True, TEXT_COLOR)
    txt_y = input_box.y + (input_box.height - txt_surface.get_height()) // 2
    screen.blit(txt_surface, (input_box.x+12, txt_y))
    # Blinking cursor
    if active and cursor_visible:
        cursor_x = input_box.x + 12 + txt_surface.get_width() + 2
        cursor_y = txt_y
        pygame.draw.line(screen, TEXT_COLOR, (cursor_x, cursor_y), (cursor_x, cursor_y+FONT.get_height()-2), 3)
    # Search button
    mouse_pos = pygame.mouse.get_pos()
    btn_color = BUTTON_HOVER if button_box.collidepoint(mouse_pos) else BUTTON_BG
    pygame.draw.rect(screen, btn_color, button_box, border_radius=12)
    pygame.draw.rect(screen, BORDER_COLOR, button_box, 2, border_radius=12)
    btn_text = FONT.render("Search", True, (255,255,255))
    btn_rect = btn_text.get_rect(center=button_box.center)
    screen.blit(btn_text, btn_rect)
    # Results label box
    pygame.draw.rect(screen, RESULT_BG, results_label_box, border_radius=12)
    pygame.draw.rect(screen, BORDER_COLOR, results_label_box, 2, border_radius=12)
    label_text = f'Search results for "{last_query}"'
    label_surface = SMALL_FONT.render(label_text, True, TEXT_COLOR)
    label_rect = label_surface.get_rect(center=results_label_box.center)
    screen.blit(label_surface, label_rect)
    # Results content box
    pygame.draw.rect(screen, RESULT_BOX_BG, results_content_box, border_radius=24)
    pygame.draw.rect(screen, BORDER_COLOR, results_content_box, 2, border_radius=24)
    # Draw results (scrollable)
    y = results_content_box.y + 20 - scroll_offset
    max_result_width = results_content_box.width - 40
    doc_spacing = 40  # Space between documents
    for line in results:
        for subline in line.split('\n'):
            wrapped_lines = wrap_text(subline, SMALL_FONT, max_result_width)
            for wline in wrapped_lines:
                if y > results_content_box.y and y < results_content_box.y + results_content_box.height - 20:
                    txt = SMALL_FONT.render(wline, True, TEXT_COLOR)
                    screen.blit(txt, (results_content_box.x+20, y))
                y += 32
        y += doc_spacing  # Add space after each document

def run_search(query):
    catcher.capture(query)
    global results, scroll_offset, last_query
    results = catcher.get_results()
    scroll_offset = 0
    last_query = query

def mainloop():
    global user_text, active, scroll_offset, cursor_visible, cursor_timer
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Title bar buttons
                if close_btn.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                if min_btn.collidepoint(event.pos):
                    pygame.display.iconify()
                # Input and button
                if input_box.collidepoint(event.pos):
                    active = True
                else:
                    active = False
                if button_box.collidepoint(event.pos):
                    run_search(user_text)
                # Scroll results
                if event.button == 4:  # scroll up
                    scroll_offset = max(0, scroll_offset - 32)
                if event.button == 5:  # scroll down
                    scroll_offset += 32
            elif event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        run_search(user_text)
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        if len(user_text) < 200 and event.unicode.isprintable():
                            user_text += event.unicode
        # Blinking cursor logic
        cursor_timer += clock.get_time()
        if cursor_timer > 500:
            cursor_visible = not cursor_visible
            cursor_timer = 0
        draw()
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mainloop()
