import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from level_generator_env import LevelGenEnv

# Конфигурация
MODEL_PATH = "level_generator.pth"
BATCH_SIZE = 2048
MEMORY_SIZE = 20000
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 0.995
EPISODES = 601
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE}")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

        """super().__init__()
        self.embed = nn.Embedding(5, 4)  # 5 состояний → 4-мерный вектор
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape) * 4, 256),  # 100 клеток * 4
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embed(x)  # [batch, 100] → [batch, 100, 4]
        x = x.view(x.size(0), -1)  # → [batch, 400]
        return self.fc(x)"""

        """super().__init__()
        self.embed = nn.Embedding(5, 4)  # 5 состояний -> 4-мерный вектор
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # входные каналы = размеру эмбеддинга
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * np.prod(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # Преобразование входных данных
        x = x.long()  # [batch, 100]

        # Применяем Embedding
        x = self.embed(x)  # [batch, 100, 4]

        # Изменение формы
        x = x.view(-1, 10, 10, 4)  # [batch, 10, 10, 4]
        x = x.permute(0, 3, 1, 2)  # [batch, 4, 10, 10]

        # Остальная часть сети
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)"""


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        self.model = DQN(self.input_shape, self.n_actions).to(DEVICE)
        self.target_model = DQN(self.input_shape, self.n_actions).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=50,
            cooldown=20
        )

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPS_START
        self.losses = []
        self.rewards = []
        self.avg_rewards = []

        self.target_update_counter = 0
        self.target_update_freq = 10

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            (state / 4.0).flatten().copy(),
            action,
            np.clip(reward, -10, 1000),
            (next_state / 4.0).flatten().copy(),
            bool(done)
        ))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            action_probs = [0.54, 0.05, 0.1, 0.3, 0.01]
            return np.random.choice(5, p=action_probs)

        state_normalized = (state.flatten() / 4.0).astype(np.float32)
        state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        dones = torch.FloatTensor([float(d) for d in dones]).to(DEVICE)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q = rewards + (1 - dones) * GAMMA * next_q
        q_values = self.model(states)  # Получаем Q-значения для всех действий

        probs = torch.softmax(q_values, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q) + 0.0001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        # self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def update_target_model(self):
        tau = 0.1  # Коэффициент смешивания
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        torch.save(self.model.state_dict(), MODEL_PATH)

    def plot_progress(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards, label='Rewards', alpha=0.3)
        plt.plot(self.avg_rewards, label='Average Rewards', linewidth=2)
        plt.title('Награды за эпизод')
        plt.subplot(1, 2, 2)
        plt.plot(self.losses)
        plt.title('Потери при обучении')
        plt.savefig('training_progress.png')
        plt.close()


def train_agent():
    # Инициализация окружения
    env = LevelGenEnv(width=10, height=10)  # Прямое создание экземпляра
    agent = DQNAgent(env)

    # Процесс обучения
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward = reward
            state = next_state
            steps += 1
            if steps % 4 == 0:
                agent.replay()

        agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY)
        agent.target_update_counter += 1
        if agent.target_update_counter % agent.target_update_freq == 0:
            agent.update_target_model()
        agent.rewards.append(total_reward)
        agent.avg_rewards.append(np.mean(agent.rewards[-100:]))

        current_lr = agent.scheduler.get_last_lr()[0]
        if episode % 10 == 0:
            print(f"Эпизод: {episode}, Награда: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, LR: {current_lr:.6f}")

        if episode % 50 == 0:
            '''test_env = LevelGenEnv(width=10, height=10)
            test_state, _ = test_env.reset()
            done = False
            while not done:
                action = agent.model(torch.FloatTensor(test_state).unsqueeze(0)).argmax().item()
                test_state, _, done, _ = test_env.step(action)'''
            fig = env.render()
            fig.savefig(f"saved_levels/episode_{episode:04d}.png")
            plt.close(fig)
            print(f"Тестовый уровень: Проходимость={env.metrics['valid']}, Путь={env.metrics['path_length']}")

        agent.scheduler.step(total_reward)

    return agent


if __name__ == '__main__':
    trained_agent = train_agent()
    trained_agent.plot_progress()
    trained_agent.save_model()
