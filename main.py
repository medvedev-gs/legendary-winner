# Симулятор торгов по "часовому" графику цен
# Импортируем необходимые библиотеки
import pygame
import sys
from scipy.stats import norm
import numpy as np
from functools import lru_cache


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
HISTORY_LENGTH = 1_000
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
INNOVATION_BUFFER_SIZE = 1000
TRADING_DAYS = 260
TRADING_HOURS = 15
PRICE_INNOVATION_SCALE = 1 / np.sqrt(TRADING_DAYS * TRADING_HOURS)

# Кнопки
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 50
BUY_BUTTON_POS = (50, 500)
SELL_BUTTON_POS = (200, 500)


class VolatilityModel:
    def __init__(self):
        self.base = BASE_VOLATILITY
        self.current = BASE_VOLATILITY
        self.speed = VOLATILITY_SPEED
        self.innovation_scale = INNOVATION_SCALE
        self.innovations = norm.rvs(scale=self.innovation_scale, size=INNOVATION_BUFFER_SIZE * 2)
        self.idx = 0

    def update(self):
        innovation = self.innovations[self.idx]
        self.idx = (self.idx + 1) % (INNOVATION_BUFFER_SIZE * 2)
        self.current = np.clip(
            self.current + self.speed * (self.base - self.current) + innovation,
            MIN_VOLATILITY,
            MAX_VOLATILITY
        )
        return self.current

    def reset(self):
        self.current = self.base


class PriceModel:
    def __init__(self) -> None:
        self.volatility_model = VolatilityModel()
        self.history_buffer = np.zeros(HISTORY_LENGTH)
        self.buffer_idx = 0
        # Инициализируем историю корректными данными
        initial_prices = self.generate_history(HISTORY_LENGTH)
        self.history_buffer[:] = initial_prices  # Заполняем буфер
        self.price = initial_prices[-1]  # Текущая цена = последнее значение

    def generate_history(self, length):
        if not HISTORY_LENGTH >= GRAPH_WIDTH:
            raise ValueError(
                f'HISTORY_LENGTH ({HISTORY_LENGTH}) должен быть >= GRAPH_WIDTH ({GRAPH_WIDTH})'
            )
        volatilities = np.full(length-1, self.volatility_model.current)
        log_returns = norm.rvs(
            scale=volatilities * PRICE_INNOVATION_SCALE,
            size=length-1
        ) + LONG_TERM_TREND
        prices = np.cumprod(np.exp(np.insert(log_returns, 0, 0))) * INITIAL_PRICE
        prices = np.maximum(prices, MIN_PRICE)
        return prices

    def update(self):
        if not HISTORY_LENGTH >= GRAPH_WIDTH:
            raise ValueError(
                f'HISTORY_LENGTH ({HISTORY_LENGTH}) должен быть >= GRAPH_WIDTH ({GRAPH_WIDTH})'
            )
        volatility = self.volatility_model.update()
        log_return = norm.rvs(scale=volatility * PRICE_INNOVATION_SCALE) + LONG_TERM_TREND
        self.price = max(MIN_PRICE, self.price * np.exp(log_return))

        # Обновляем историю
        self.history_buffer[self.buffer_idx] = self.price
        self.buffer_idx = (self.buffer_idx + 1) % HISTORY_LENGTH

        # Получаем видимую историю (последние GRAPH_WIDTH значений)
        if self.buffer_idx >= GRAPH_WIDTH:
            self.visible_history = self.history_buffer[self.buffer_idx - GRAPH_WIDTH : self.buffer_idx]
        else:
            # Если буфер "перевернулся", соединяем конец и начало
            self.visible_history = np.concatenate([
                self.history_buffer[HISTORY_LENGTH - (GRAPH_WIDTH - self.buffer_idx):],
                self.history_buffer[:self.buffer_idx]
            ])
        
        return self.price

class StockVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    @lru_cache(maxsize=50)
    def render_text(self, text, color, size):
        font = pygame.font.SysFont('Arial', size)
        return font.render(text, True, color)

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

    def draw_graph(self, history):
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

    def draw_ui(self, cash, shares, price, volatility):
        # Кнопки
        buy_button = pygame.draw.rect(self.screen, GREEN, (*BUY_BUTTON_POS, BUTTON_WIDTH, BUTTON_HEIGHT))
        sell_button = pygame.draw.rect(self.screen, RED, (*SELL_BUTTON_POS, BUTTON_WIDTH, BUTTON_HEIGHT))
        
        # Текст
        texts = [
            self.render_text(f'Деньги: ${cash:.2f}', BLACK, 14),
            self.render_text(f'Акции: {shares}', BLACK, 14),
            self.render_text(f'Цена: ${price:.2f}', BLACK, 14),
            self.render_text(f'Волатильность: {volatility:.2f}', RED, 14)
        ]
        
        for i, text in enumerate(texts):
            self.screen.blit(text, (TEXT_START_X + i * TEXT_SPACING, TEXT_START_Y))
            
        return buy_button, sell_button


def main():
    pygame.init()
    visualizer = StockVisualizer()
    price_model = PriceModel()
    clock = pygame.time.Clock()
    
    # Игровые параметры
    cash = INITIAL_CASH
    shares = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                buy_button, sell_button = visualizer.draw_ui(cash, shares, price_model.price, price_model.volatility_model.current)
                
                if buy_button.collidepoint(mouse_pos) and cash >= price_model.price:
                    shares += 1
                    cash -= price_model.price
                    
                elif sell_button.collidepoint(mouse_pos) and shares > 0:
                    shares -= 1
                    cash += price_model.price
        
        # Отрисовка:
        price_model.update()

        visualizer.draw_background()  # Сначала фон и сетка
        visualizer.draw_graph(price_model.visible_history)
        visualizer.draw_ui(cash, shares, price_model.price, price_model.volatility_model.current)
    
        pygame.display.flip()
        clock.tick(FPS)
 
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
