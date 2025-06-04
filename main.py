# Импортируем необходимые библиотеки
import pygame
import sys
from scipy.stats import binom, norm
import numpy as np


# ===== КОНСТАНТЫ =====
# Окно
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 4  # Количество обновлений в секунду

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# График
GRID_COLOR = (220, 220, 220)  # Светло-серый для сетки
BACKGROUND_COLOR = (210, 210, 210)  # Серый фон как в сапёре
PRICE_LABELS_COLOR = (100, 100, 100)  # Цвет меток цен
PRICE_LABELS_COUNT = 5  # Количество меток цен на графике
GRAPH_HEIGHT = 400
GRAPH_TOP = 50
GRAPH_LEFT = 50
GRAPH_WIDTH = SCREEN_WIDTH - 2 * GRAPH_LEFT  # 800 - 2*50 = 700
HISTORY_LENGTH = GRAPH_WIDTH  # Теперь они равны
TEXT_START_X = 50  # Начальная позиция текста по X
TEXT_START_Y = 450  # Позиция текста по Y
TEXT_SPACING = 150

# Акции
INITIAL_PRICE = 50.0
INITIAL_CASH = 1000.0
BINOM_N = 100  # Количество испытаний в биномиальном распределении
BINOM_P = 0.5  # Вероятность успеха
PRICE_CHANGE_SCALE = 5

# Волатильность
BASE_VOLATILITY = 1.2
VOLATILITY_SPEED = 0.1
INNOVATION_SCALE = 0.3
MIN_VOLATILITY = 0.1
INNOVATION_BUFFER_SIZE = 1000

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
        self.innovations = norm.rvs(scale=self.innovation_scale, size=INNOVATION_BUFFER_SIZE)
        self.idx = 0

    def update(self):
        innovation = self.innovations[self.idx % INNOVATION_BUFFER_SIZE]
        self.idx += 1
        self.current = max(
            MIN_VOLATILITY,
            self.current + self.speed * (self.base - self.current) + innovation
        )
        return self.current

    def reset(self):
        self.current = self.base


class PriceModel:
    def __init__(self):
        self.price = INITIAL_PRICE
        self.volatility_model = VolatilityModel()
        self.binom_scale = 1 / np.sqrt(BINOM_N * BINOM_P * (1 - BINOM_P))
        self.full_history = self.generate_history(HISTORY_LENGTH)
        self.visible_history = list(self.full_history[-GRAPH_WIDTH:])

    def generate_history(self, length):
        """Генерирует историю цен с реалистичными колебаниями"""
        changes = binom(n=BINOM_N, p=BINOM_P).rvs(size=length-1) - BINOM_N * BINOM_P
        changes = changes * (self.volatility_model.base / PRICE_CHANGE_SCALE)
        history = np.maximum(1.0, self.price + np.cumsum(changes))
        return np.concatenate([[self.price], history])
    
    def update(self):
        volatility = self.volatility_model.update()
        change = binom(BINOM_N, BINOM_P).rvs() - BINOM_N * BINOM_P
        self.price = max(1.0, self.price + change * volatility * self.binom_scale)
    
        # Обновляем полную историю
        self.full_history = np.roll(self.full_history, -1)
        self.full_history[-1] = self.price
        if len(self.full_history) > HISTORY_LENGTH:
            self.full_history = self.full_history[1:]
    
        # Обновляем видимую часть
        self.visible_history = list(self.full_history[-GRAPH_WIDTH:])
        return self.price


class StockVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 12)  # Для меток цен
        self.price_labels_font = pygame.font.SysFont('Arial', 10)  # Для мелкого текста

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
            label = self.small_font.render(f"{price:.2f}", True, PRICE_LABELS_COLOR)
            self.screen.blit(label, (label_x, y - 10))

    def draw_graph(self, history):
        if not history:
            return
            
        history = np.array(history)
        min_p, max_p = history.min(), history.max()
        
        # Метки цен
        self.draw_price_labels(min_p, max_p)
        
        # Масштабирование
        if max_p == min_p:
            scaled = np.full(len(history), GRAPH_TOP + GRAPH_HEIGHT // 2, dtype=int)
        else:
            scaled = GRAPH_TOP + GRAPH_HEIGHT - ((history - min_p) / (max_p - min_p) * GRAPH_HEIGHT).astype(int)

        # Линия графика
        points = [(GRAPH_LEFT + i, scaled[i]) for i in range(len(history))]
        if len(points) > 1:
            pygame.draw.lines(self.screen, BLUE, False, points, 2)

    def draw_ui(self, cash, shares, price, volatility):
        # Кнопки
        buy_button = pygame.draw.rect(self.screen, GREEN, (*BUY_BUTTON_POS, BUTTON_WIDTH, BUTTON_HEIGHT))
        sell_button = pygame.draw.rect(self.screen, RED, (*SELL_BUTTON_POS, BUTTON_WIDTH, BUTTON_HEIGHT))
        
        # Текст
        texts = [
            self.font.render(f'Деньги: ${cash:.2f}', True, BLACK),
            self.font.render(f'Акции: {shares}', True, BLACK),
            self.font.render(f'Цена: ${price:.2f}', True, BLACK),
            self.font.render(f'Волатильность: {volatility:.2f}', True, RED)
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
