import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class LevelGenEnv(gym.Env):
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height

        # Определение пространства действий и состояний
        self.action_space = spaces.Discrete(5)  # 0-4: add_platform, add_enemy, add_coin, remove, finish
        self.observation_space = spaces.Box(low=0, high=4, shape=(height, width), dtype=np.int32)

        # Конфигурация наград
        self.reward_config = {
            "valid_level": 200.0,
            "invalid_penalty": 50.0,
            "coin_weight": 1.3,
            "enemy_weight": 1.1,
            "coin_penalty": 1.8,
            "enemy_penalty": 1.5,
            "max_coins": 8,
            "max_enemies": 5,
            "path_length_reward": 2,
            "completion_bonus": 50.0,
            "platform": 1,
            "diversity_bonus": 50,
            'change_penalty': 0.1
        }

        # Инициализация уровня
        self.level = np.zeros((height, width), dtype=np.int32)
        self.player_pos = (0, 0)
        self.level[self.player_pos] = 4  # Размещаем игрока

        # Метрики уровня
        self.metrics = {
            "path_length": 0,
            "coins": 0,
            "enemies": 0,
            "valid": False,
            "diversity": 0.0
        }

        # Координаты начала и конца
        self.start_y, self.start_x = np.argwhere(self.level == 4)[0]
        self.end_x, self.end_y = self.width - 1, self.height - 1  # Правая нижняя клетка
        self.grid = np.zeros((height, width), dtype=int)

        self.step_count = 0
        self.max_steps = 300

        self.fig, self.ax = plt.subplots()
        self.colors = {
            0: 'white',  # Пустое пространство
            1: 'brown',  # Платформа
            2: 'black',  # Враг
            3: 'yellow',  # Монета
            4: 'green',  # Игрок
        }

    def reset(self, **kwargs):
        """Сброс среды к начальному состоянию"""
        self.level = np.zeros((self.height, self.width), dtype=np.int32)
        self.player_pos = (0, 0)
        self.level[self.player_pos] = 4  # Игрок

        # Возвращаем начальное состояние и пустой info
        return self.level.copy(), {}

    def step(self, action):
        """Выполнение действия и возврат нового состояния"""
        old_state = self.level.copy()
        reward = self._get_reward(old_state)

        if isinstance(action, torch.Tensor):
            action = action.item()

        done = False  # Эпизод продолжается

        # Применение действия
        self.step_count += 1
        if int(action) == 4 or self._is_level_passable():  # Специальное действие "завершить уровень"
            done = True
            self.step_count = 0
            if self._is_level_complete():
                reward += self.reward_config["completion_bonus"]
            elif not self._is_level_passable():
                reward -= self.reward_config["invalid_penalty"]
        else:
            # Добавление/удаление объектов
            x, y = self._random_position()
            if action == 0 and self.level[y, x] == 0:
                self.level[y, x] = 1  # Платформа
                if self._get_path_length(x, y) > 0:
                    reward += 0.5
            elif action == 1 and self.level[y, x] == 0:
                self.level[y, x] = 2  # Враг
            elif action == 2 and self.level[y, x] == 0:
                self.level[y, x] = 3  # Монета
                if self._get_path_length(x, y) > 0:
                    reward += 0.5
            elif action == 3 and self.level[y, x] != 4:
                self.level[y, x] = 0  # Удаление

        # Лимит шагов для предотвращения бесконечных циклов
        if self.step_count >= self.max_steps:
            done = True
            self.step_count = 0

        # Вычисление метрик
        self._calculate_metrics()
        reward = self._get_reward(old_state)

        # Информация для отладки
        '''print(
            f"Step: {self.step_count} | "
            f"Action: {action} | "
            f"Reward: {reward:.1f} | "
            f"Valid: {self.metrics['valid']} | "
            f"Platforms: {np.sum(self.level == 1)} | "
            f"Coins: {np.sum(self.level == 3)}"
        )'''

        return self.level.copy(), reward, done, False

    def _random_position(self):
        """Генерация случайной позиции на уровне"""
        return np.random.randint(0, self.width), np.random.randint(0, self.height)

    def _calculate_metrics(self):
        """Вычисление всех метрик уровня"""
        self.metrics["coins"] = np.sum(self.level == 3)
        self.metrics["enemies"] = np.sum(self.level == 2)
        self.metrics["valid"] = self._is_level_passable()
        self.metrics["path_length"] = self._get_path_length(self.end_x, self.end_y) if self.metrics["valid"] else 0
        self.metrics["diversity"] = self._calculate_diversity()

    def _get_reward(self, old_state):
        """Вычисление награды на основе текущего состояния"""
        cfg = self.reward_config
        reward = 0

        # Штраф за изменения
        reward -= np.sum(np.abs(self.level - old_state)) * cfg['change_penalty']

        # Награды за валидность уровня
        if self.metrics["valid"]:
            reward += cfg["valid_level"]
            # Бонус за плотность платформ
            reward += 50 * np.clip(np.sum(self.level == 1), 0, 12) / 12

            # Бонус за разнообразие элементов
            reward += self.metrics["diversity"] * cfg["diversity_bonus"]

        # Награда за монеты (с ограничением)
        reward += min(self.metrics["coins"], cfg["max_coins"]) * cfg["coin_weight"]
        coin_penalty = max(0, self.metrics["coins"] - cfg["max_coins"]) * cfg["coin_penalty"]
        reward -= coin_penalty

        # Награда за врагов (с ограничением)
        reward += min(self.metrics["enemies"], cfg["max_enemies"]) * cfg["enemy_weight"]
        enemy_penalty = max(0, self.metrics["enemies"] - cfg["max_enemies"]) * cfg["enemy_penalty"]
        reward -= enemy_penalty

        # Бонус за платформы
        reward += np.sum(self.level == 1) * cfg["platform"]

        # Бонус за длинные пути
        reward += self.metrics["path_length"] * cfg["path_length_reward"]

        return reward

    def _get_path(self, x, y):
        """Вычисление длины кратчайшего пути"""
        # Только платформы (1), монеты (3) и игрок (4) считаются проходимыми
        walkable = np.isin(self.level, [1, 3, 4])
        grid = Grid(matrix=walkable.astype(int))

        # Проверка наличия игрока
        start_nodes = np.argwhere(self.level == 4)
        if len(start_nodes) == 0:
            return 0

        # Поиск пути с обработкой ошибок
        try:
            path, _ = AStarFinder().find_path(
                grid.node(self.start_x, self.start_y),
                grid.node(x, y),
                grid
            )
            return path if len(path) > 0 else 0
        except Exception as e:
            return 0

    def _get_path_length(self, x, y):
        return len(self._get_path(x, y)) if self._get_path(x, y) != 0 else 0

    def _is_level_passable(self):
        """Проверка проходимости уровня"""
        # Проверка наличия игрока
        if np.sum(self.level == 4) != 1:
            return False

        # Проверка пути
        return True if (self._get_path_length(self.end_x, self.end_y) > 0) else False

    def _is_level_complete(self):
        """Проверка наличия всех необходимых элементов"""
        return (
                np.sum(self.level == 1) >= 10 and  # Минимум 10 платформ
                np.sum(self.level == 2) >= 1 and  # Минимум 1 враг
                np.sum(self.level == 3) >= 2 and  # Минимум 2 монеты
                self._is_level_passable()  # Уровень проходим
        )

    def _calculate_diversity(self):
        """Вычисление метрики разнообразия элементов"""
        counts = np.bincount(self.level.flatten(), minlength=5)
        proportions = counts / np.sum(counts)
        entropy = -np.sum(proportions * np.log(proportions + 1e-9))
        return entropy/np.log(5)  # Энтропия

    '''def render(self, show_path=True):
        # Создаем матрицу для визуализации
        vis_grid = np.copy(self.level)
        vis_grid[self.start_x, self.start_y] = 3  # Старт (x,y → row,col)
        vis_grid[self.end_x, self.end_y] = 4      # Финиш

        # Отрисовка сетки
        self.ax.clear()
        self.ax.imshow(vis_grid, cmap='viridis', vmin=0, vmax=4)

        # Добавление пути (если требуется)
        if show_path and self._is_level_passable():
            path = self._get_path()
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            self.ax.plot(path_x, path_y, 'r-', linewidth=2)

        plt.pause(0.01)'''

    def render(self):
        """Визуализирует текущее состояние уровня с использованием цветов из self.colors."""
        # Проверяем, существует ли окно, и пересоздаем при необходимости
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots()

        # Создаем RGB-массив
        height, width = self.level.shape
        rgb_array = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                cell_value = int(self.level[y, x])
                color_name = self.colors.get(cell_value)
                rgb = mcolors.to_rgb(color_name)
                rgb_array[y, x] = rgb

        # Очищаем оси и отображаем новый уровень
        self.ax.clear()
        self.ax.imshow(rgb_array)

        # Настройка сетки
        self.ax.set_xticks(np.arange(-0.5, width), minor=True)
        self.ax.set_yticks(np.arange(-0.5, height), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        self.ax.tick_params(which='minor', length=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Обновление графика
        plt.draw()
        plt.pause(0.001)

        return self.fig
