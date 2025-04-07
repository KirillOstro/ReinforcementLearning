import gym
from gym import spaces
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class LevelGenEnv(gym.Env):
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(5)  # 0-4: add_platform, add_enemy, add_coin, remove, skip
        self.observation_space = spaces.Box(low=0, high=4, shape=(height, width), dtype=np.int32)
        self.level = np.zeros((height, width), dtype=np.int32)
        self.player_pos = (0, 0)  # Стартовая позиция игрока
        self.level[0][0] = 4  # Игрок

    def step(self, action):
        if action == 0:  # Добавить платформу
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.level[y][x] == 0:
                self.level[y][x] = 1
        elif action == 1:  # Добавить врага
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.level[y][x] == 0:
                self.level[y][x] = 2
        elif action == 2:  # Добавить монету
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.level[y][x] == 0:
                self.level[y][x] = 3
        elif action == 3:  # Удалить объект
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.level[y][x] != 4:  # Не удаляем игрока
                self.level[y][x] = 0

        # Проверка проходимости (A* алгоритм)
        reward = self._calculate_reward()
        done = (action == 4)  # Завершить уровень
        return self.level.copy(), reward, done, {}

    def _calculate_reward(self):
        # Проверка пути от игрока до выхода (правый нижний угол)
        grid = Grid(matrix=(self.level != 2).astype(int))  # Враги — препятствия
        start = grid.node(self.player_pos[1], self.player_pos[0])
        end = grid.node(self.width - 1, self.height - 1)
        finder = AStarFinder()
        path, _ = finder.find_path(start, end, grid)

        if path:
            coin_reward = 0.5 * np.sum(self.level == 3)
            return 10 + coin_reward - np.sum(self.level != 0) * 0.1
        else:
            return -5

    def reset(self):
        self.level = np.zeros((self.height, self.width), dtype=np.int32)
        self.level[0][0] = 4  # Игрок
        return self.level.copy()

    def render(self, mode='human'):
        symbols = {0: '.', 1: '=', 2: 'E', 3: 'C', 4: 'P'}
        for row in self.level:
            print(' '.join([symbols[cell] for cell in row]))
        print('---')