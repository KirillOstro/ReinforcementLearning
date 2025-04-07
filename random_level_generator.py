import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from level_generator_env import LevelGenEnv

# Настройка стиля
plt.style.use('dark_background')


class LevelVisualizer:
    def __init__(self):
        # Цветовая схема и подписи
        self.cmap = plt.cm.viridis
        self.labels = {
            0: 'Пустота',
            1: 'Платформа',
            2: 'Враг',
            3: 'Монета',
            4: 'Игрок'
        }
        self.colors = {
            0: [0.9, 0.9, 0.9],  # Серый
            1: [0.5, 0.3, 0.1],  # Коричневый
            2: [1, 0, 0],  # Красный
            3: [1, 1, 0],  # Желтый
            4: [0, 0.5, 1]  # Синий
        }

    def render_frame(self, state, ax):
        """Рендер кадра с подписями"""
        img = np.zeros((*state.shape, 3))
        for y in range(state.shape[0]):
            for x in range(state.shape[1]):
                img[y, x] = self.colors[state[y, x]]

        ax.imshow(img)

        # Добавляем подписи объектов
        for y in range(state.shape[0]):
            for x in range(state.shape[1]):
                if state[y, x] != 0:  # Не подписываем пустые клетки
                    ax.text(x, y, self.labels[state[y, x]],
                            ha='center', va='center',
                            color='white' if state[y, x] in [1, 2] else 'black',
                            fontsize=8)

        # Легенда
        legend_elements = [Patch(facecolor=color, label=name)
                           for name, color in zip(self.labels.values(),
                                                  self.colors.values())]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                  loc='upper left')


# Генерация кадров
env = LevelGenEnv(width=8, height=8)
visualizer = LevelVisualizer()
frames = []

env.reset()
for _ in range(10):
    action = env.action_space.sample()
    state, _, done, _ = env.step(action)
    frames.append(state.copy())
    if done:
        break

# Создание анимации
fig, ax = plt.subplots(figsize=(10, 8))
fig.subplots_adjust(right=0.8)  # Место для легенды


def update(i):
    ax.clear()
    ax.set_title(f"Генерация уровня (шаг {i + 1}/{len(frames)})")
    visualizer.render_frame(frames[i], ax)
    ax.axis('off')


anim = FuncAnimation(fig, update, frames=len(frames), interval=800)

# Сохранение GIF
writer = PillowWriter(fps=1.5, bitrate=1000)
anim.save("level_generation.gif", writer=writer, dpi=120)
plt.close()

print("Анимация сохранена как 'level_generation.gif'")