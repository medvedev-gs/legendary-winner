# Симулятор торгов по "часовому" графику цен
# Импортируем необходимые библиотеки
import pygame
import sys
from dataclasses import dataclass
from scipy.stats import norm
import numpy as np
from functools import lru_cache
from typing import List, Dict, Union, Tuple


# ===== КОНСТАНТЫ =====
# Окно
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 1  # Количество обновлений в секунду

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PANEL_COLOR = (110, 110, 110)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)

# График
GRID_COLOR = (220, 220, 220)  # Светло-серый для сетки
BACKGROUND_COLOR = (210, 210, 210)  # Серый фон
PRICE_LABELS_COLOR = (100, 100, 100)  # Цвет меток цен
PRICE_LABELS_COUNT = 5  # Количество меток цен на графике
GRAPH_HEIGHT = 400
GRAPH_TOP = 50
GRAPH_LEFT = 50
GRAPH_WIDTH = SCREEN_WIDTH - 2 * GRAPH_LEFT  # 800 - 2*50 = 700
TEXT_START_X = 50  # Начальная позиция текста по X
TEXT_START_Y = 450  # Позиция текста по Y
TEXT_SPACING = 150

# Акции
HISTORY_LENGTH = 2000
INITIAL_PRICE = 100.0
INITIAL_CASH = 1000.0
MIN_PRICE = 0.01
LONG_TERM_TREND = 0.0001

# Волатильность
BASE_VOLATILITY = 0.15
VOLATILITY_SPEED = 0.05
INNOVATION_SCALE = 0.1
MIN_VOLATILITY = 0.1
MAX_VOLATILITY = 2.5
INNOVATION_BUFFER_SIZE = 2000
TRADING_DAYS = 260
TRADING_HOURS = 15
PRICE_INNOVATION_SCALE = 1 / np.sqrt(TRADING_DAYS * TRADING_HOURS)

# Кнопки
BUTTON_WIDTH, BUTTON_HEIGHT = 50, 50
BUY_BUTTON_POS = (50, 500)
SELL_BUTTON_POS = (200, 500)


@dataclass(frozen=True, slots=True)
class GameMode:
    name: str
    description: str
    settings: Dict[str, Union[int, float]]


class GameModeSystem:
    def __init__(self):
        self.modes: List[GameMode] = [
            GameMode(
                name="Новичок",
                description="Низкая волатильность",
                settings={"speed": 1, "volatility_mult": 0.7, "commission": 0, "initial_cash": 10000, "trend_strength": 0.5}
            ),
            GameMode(
                name="Турбо - режим",
                description="Высокая скорость",
                settings={"speed": 3, "volatility_mult": 1.5, "commission": 0.001, "initial_cash": 5000, "trend_strength": 1.2}
            ),
            GameMode(
                name="Крипто - Анархия",
                description="Экстремальная волатильность",
                settings={"speed": 2, "volatility_mult": 2.5, "commission": 0.002, "initial_cash": 2000, "trend_strength": 2.0})
        ]
        self.current_mode: int = 0

    def get_current(self) -> GameMode:
        return self.modes[self.current_mode]

    def next_mode(self) -> GameMode:
        self.current_mode = (self.current_mode + 1) % len(self.modes)
        return self.get_current()


class VolatilityModel:
    def __init__(self):
        self.base = BASE_VOLATILITY
        self.current = BASE_VOLATILITY
        self.speed = VOLATILITY_SPEED
        self.innovation_scale = INNOVATION_SCALE
        self.innovations = norm.rvs(scale=self.innovation_scale, size=INNOVATION_BUFFER_SIZE)
        self.idx = 0

    def update(self) -> float:
        innovation = self.innovations[self.idx]
        self.idx = (self.idx + 1) % (INNOVATION_BUFFER_SIZE)
        self.current = np.clip(
            self.current + self.speed * (self.base - self.current) + innovation,
            MIN_VOLATILITY,
            MAX_VOLATILITY
        )
        return self.current

    def reset(self) -> None:
        self.current = self.base


class PriceModel:
    def __init__(self, game_mode_system: GameModeSystem) -> None:
        self.mode_system = game_mode_system
        self.volatility_model = VolatilityModel()
        self.visible_history = np.zeros(GRAPH_WIDTH)
        initial_prices = self.generate_history(HISTORY_LENGTH)
        self.visible_history[:] = initial_prices[-GRAPH_WIDTH:]
        self.price = initial_prices[-1]

    def generate_history(self, length: int):
        if not HISTORY_LENGTH >= GRAPH_WIDTH:
            raise ValueError(
                f'HISTORY_LENGTH ({HISTORY_LENGTH}) должен быть >= GRAPH_WIDTH ({GRAPH_WIDTH})'
            )
        volatilities = np.array([self.volatility_model.update() for _ in range(length-1)])
        log_returns = norm.rvs(
            scale=volatilities * PRICE_INNOVATION_SCALE,
            size=length-1
        ) + LONG_TERM_TREND
        prices = np.cumprod(np.exp(np.insert(log_returns, 0, 0))) * INITIAL_PRICE
        prices = np.maximum(prices, MIN_PRICE)
        return prices

    def update(self) -> float:
        if not HISTORY_LENGTH >= GRAPH_WIDTH:
            raise ValueError(
                f'HISTORY_LENGTH ({HISTORY_LENGTH}) должен быть >= GRAPH_WIDTH ({GRAPH_WIDTH})'
            )
        mode = self.mode_system.get_current()
        volatility = self.volatility_model.update() * mode.settings["volatility_mult"]
        trend = LONG_TERM_TREND * mode.settings["trend_strength"]
        log_return = norm.rvs(scale=volatility * PRICE_INNOVATION_SCALE) + trend
        self.price = max(MIN_PRICE, self.price * np.exp(log_return))

        # Получаем видимую историю (последние GRAPH_WIDTH значений)
        if GRAPH_WIDTH <= 1:
            raise ValueError(f'GRAPH_WIDTH должен быть некоторой ширины')
        self.visible_history[:-1] = self.visible_history[1:]
        self.visible_history[-1] = self.price
        return self.price

class StockVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    @lru_cache(maxsize=50)
    def render_text(self, text: str, color: Tuple[int, int, int], size: int):
        font = pygame.font.SysFont('Arial', size)
        return font.render(text, True, color)

    def draw_mode_selection(self, mode_system: GameModeSystem):
        """Отрисовка экрана выбора режима"""
        self.screen.fill(PANEL_COLOR)
    
        title = self.render_text("ВЫБЕРИТЕ РЕЖИМ ИГРЫ", DARK_GRAY, 32)
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 50))
    
        mode = mode_system.get_current()
    
        # Отображение текущего режима
        mode_rect = pygame.Rect(SCREEN_WIDTH//2 - 200, 150, 400, 300)
        pygame.draw.rect(self.screen, WHITE, mode_rect, 0, 15)
        pygame.draw.rect(self.screen, BLUE, mode_rect, 3, 15)
    
        # Название режима
        name_text = self.render_text(mode.name, BLUE, 28)
        self.screen.blit(name_text, (SCREEN_WIDTH//2 - name_text.get_width()//2, 170))
    
        # Описание
        desc_text = self.render_text(mode.description, DARK_GRAY, 18)
        self.screen.blit(desc_text, (SCREEN_WIDTH//2 - desc_text.get_width()//2, 220))

        # Параметры (проверяем наличие ключей в settings)
        params = [
            f"Скорость: {mode.settings.get('speed', 1)}x",
            f"Волатильность: {mode.settings.get('volatility_mult', 1)}x",
            f"Комиссия: {mode.settings.get('commission', 0)*100:.1f}%",
            f"Начальный капитал: ${mode.settings.get('initial_cash', 1000):,}"
        ]

        for i, param in enumerate(params):
            param_text = self.render_text(param, BLACK, 16)
            self.screen.blit(param_text, (SCREEN_WIDTH//2 - param_text.get_width()//2, 300 + i*25))

        # Кнопки
        next_btn = pygame.draw.rect(self.screen, LIGHT_GRAY, (SCREEN_WIDTH//2 + 50, 450, 120, 40), 0, 10)
        next_text = self.render_text("Далее →", BLACK, 16)
        self.screen.blit(next_text, (SCREEN_WIDTH//2 + 110 - next_text.get_width()//2, 460))
    
        prev_btn = pygame.draw.rect(self.screen, LIGHT_GRAY, (SCREEN_WIDTH//2 - 170, 450, 120, 40), 0, 10)
        prev_text = self.render_text("← Назад", BLACK, 16)
        self.screen.blit(prev_text, (SCREEN_WIDTH//2 - 110 - prev_text.get_width()//2, 460))
    
        start_btn = pygame.draw.rect(self.screen, GREEN, (SCREEN_WIDTH//2 - 100, 510, 200, 50), 0, 10)
        start_text = self.render_text("НАЧАТЬ ТОРГОВЛЮ", WHITE, 18)
        self.screen.blit(start_text, (SCREEN_WIDTH//2 - start_text.get_width()//2, 525))
    
        return prev_btn, next_btn, start_btn

    def draw_background(self):
        """Рисует фон и сетку"""
        # Заливка фона
        self.screen.fill(BACKGROUND_COLOR)
        
        # Сетка
        for y in range(GRAPH_TOP, GRAPH_TOP + GRAPH_HEIGHT + 1, GRAPH_HEIGHT // 10):
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (GRAPH_LEFT, y), 
                           (GRAPH_LEFT + GRAPH_WIDTH, y), 1)
        
        # Рамка графика
        pygame.draw.rect(self.screen, BLACK, 
                        (GRAPH_LEFT, GRAPH_TOP, GRAPH_WIDTH, GRAPH_HEIGHT), 1)

    def draw_price_labels(self, min_price, max_price):
        """Рисует метки цен справа от графика"""
        label_x = GRAPH_LEFT + GRAPH_WIDTH + 10
        step = GRAPH_HEIGHT / (PRICE_LABELS_COUNT - 1)
        
        for i in range(PRICE_LABELS_COUNT):
            y = GRAPH_TOP + i * step
            price = max_price - (max_price - min_price) * (i / (PRICE_LABELS_COUNT - 1))
            label = self.render_text(f"{price:.2f}", BLACK, 11)
            self.screen.blit(label, (label_x, y - 10))

    def draw_graph(self, history: np.ndarray):
        if len(history) == 0:
            return

        min_p, max_p = history.min(), history.max()
        
        # Метки цен
        self.draw_price_labels(min_p, max_p)
        
        # Масштабирование
        if max_p == min_p:
            scaled = np.full(len(history), GRAPH_TOP + GRAPH_HEIGHT // 2, dtype=int)
        else:
            scaled = GRAPH_TOP + GRAPH_HEIGHT - ((history - min_p) / (max_p - min_p) * GRAPH_HEIGHT).astype(int)

        # Линия графика
        x_coords = np.arange(GRAPH_LEFT, GRAPH_LEFT + len(history))
        points = np.column_stack((x_coords, scaled)).astype(int)
        pygame.draw.lines(self.screen, BLUE, False, points, 2)

    def draw_ui(self, cash, shares, price, volatility, mode_system: GameModeSystem):
        # Кнопки
        buy_button = pygame.draw.rect(self.screen, GREEN, (*BUY_BUTTON_POS, BUTTON_WIDTH, BUTTON_HEIGHT))
        sell_button = pygame.draw.rect(self.screen, RED, (*SELL_BUTTON_POS, BUTTON_WIDTH, BUTTON_HEIGHT))
        
        # Текст
        mode = mode_system.get_current()
        texts = [
            self.render_text(f'Деньги: ${cash:.2f}', BLACK, 14),
            self.render_text(f'Акции: {shares}', BLACK, 14),
            self.render_text(f'Цена: ${price:.2f}', BLACK, 14),
            self.render_text(f'Режим: {mode.name}', BLACK, 14)
        ]
        
        for i, text in enumerate(texts):
            self.screen.blit(text, (TEXT_START_X + i * TEXT_SPACING, TEXT_START_Y))
            
        return buy_button, sell_button

def main():
    pygame.init()
    visualizer = StockVisualizer()
    mode_system = GameModeSystem()
    clock = pygame.time.Clock()
    
    game_state = "mode_selection"
    running = True
    mode = None
    
    while running:
        if game_state == "trading":
            mode = mode_system.get_current()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if game_state == "mode_selection":
                    prev_btn, next_btn, start_btn = visualizer.draw_mode_selection(mode_system)
                    
                    if next_btn.collidepoint(mouse_pos):
                        mode_system.next_mode()
                    elif prev_btn.collidepoint(mouse_pos):
                        mode_system.next_mode()
                    elif start_btn.collidepoint(mouse_pos):
                        game_state = "trading"
                        price_model = PriceModel(mode_system)
                        cash = mode_system.get_current().settings["initial_cash"]
                        shares = 0
                
                elif game_state == "trading":
                    buy_button, sell_button = visualizer.draw_ui(
                        cash, shares, price_model.price, 
                        price_model.volatility_model.current, mode_system
                    )
                    
                    if buy_button.collidepoint(mouse_pos) and cash >= price_model.price:
                        shares += 1
                        cash -= price_model.price * (1 + mode.settings["commission"])
                        
                    elif sell_button.collidepoint(mouse_pos) and shares > 0:
                        shares -= 1
                        cash += price_model.price * (1 - mode.settings["commission"])

        # Отрисовка
        if game_state == "mode_selection":
            visualizer.draw_mode_selection(mode_system)
        elif game_state == "trading":
            price_model.update()
            visualizer.draw_background()
            visualizer.draw_graph(price_model.visible_history)
            visualizer.draw_ui(
                cash, shares, price_model.price, 
                price_model.volatility_model.current, mode_system
            )
        
        pygame.display.flip()
        clock.tick(FPS * (mode.settings["speed"] if game_state == "trading" and mode else 1))

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
