import torch
import numpy as np
import matplotlib.pyplot as plt
from train_model import DQN
from level_generator_env import LevelGenEnv

MODEL_PATH = "level_generator.pth"


class LevelGenerator:
    def __init__(self):
        self.env = LevelGenEnv(width=10, height=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

    def generate_level(self):
        for i in range(10):
            state, _ = self.env.reset()
            done = False
            while not done:
                state_normalized = (state.flatten() / 4.0).astype(np.float32)
                state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.model(state_tensor).argmax().item()

                # Выполняем действие в среде
                next_state, _, done, _ = self.env.step(action)
                state = next_state

            fig = self.env.render()
            fig.savefig(f"saved_levels/!TestLevel{i+1}.png")
            plt.close(fig)
            print(f"Тестовый уровень {i+1}: Проходимость={self.env.metrics['valid']}, Путь={self.env.metrics['path_length']}")


if __name__ == "__main__":
    generator = LevelGenerator()
    generator.generate_level()
