# scripts/test_manual_ppo.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import tarware
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "tarware-tiny-3agvs-2pickers-partialobs-v1"
MODEL_DIR = "outputs/manual_ppo_models"  # carpeta donde guardaste los modelos
NUM_EPISODES = 5
MAX_STEPS = 200
RENDER = True  # Cambia a True si quieres ver el entorno

# -----------------------------
# Actor Network (igual que en entrenamiento)
# -----------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.action_head(x), dim=-1)

# -----------------------------
# Inicializar entorno
# -----------------------------
env = gym.make(ENV_NAME)
obs_init = env.reset()
NUM_AGENTS = len(obs_init)
print(f"Numero de agentes: {NUM_AGENTS}")

# -----------------------------
# Cargar actores
# -----------------------------
actors = []
for i in range(NUM_AGENTS):
    obs_dim = obs_init[i].shape[0]
    act_dim = env.action_space[i].n
    actor = Actor(obs_dim, act_dim).to(DEVICE)
    actor.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"actor_agent{i}.pt"), map_location=DEVICE))
    actor.eval()  # Modo evaluación
    actors.append(actor)

# -----------------------------
# Función para seleccionar acción
# -----------------------------
def select_action(actor, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    probs = actor(obs_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item()

# -----------------------------
# Test del modelo
# -----------------------------
for episode in range(NUM_EPISODES):
    obs = env.reset()
    episode_rewards = np.zeros(NUM_AGENTS)

    for step in range(MAX_STEPS):
        if RENDER:
            env.render()
            time.sleep(0.1)

        actions = [select_action(actors[i], obs[i]) for i in range(NUM_AGENTS)]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += np.array(rewards)

        dones = [t or tr for t, tr in zip(terminated, truncated)]
        obs = next_obs

        if all(dones):
            break

    print(f"Episode {episode+1}, Rewards: {episode_rewards}")

env.close()


