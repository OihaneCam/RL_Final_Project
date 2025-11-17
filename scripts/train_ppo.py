# # train_ppo.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import gymnasium as gym
# import tarware

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ENV_NAME = "tarware-tiny-3agvs-2pickers-partialobs-v1"
# NUM_EPISODES = 500
# MAX_STEPS = 200
# GAMMA = 0.99
# LR = 1e-3

# # Actor-Critic network
# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.action_head = nn.Linear(128, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         action_logits = self.action_head(x)
#         return F.softmax(action_logits, dim=-1)

# class Critic(nn.Module):
#     def __init__(self, obs_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.value_head = nn.Linear(128, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         value = self.value_head(x)
#         return value

# # Initialize environment
# env = gym.make(ENV_NAME)
# obs = env.reset()  # obs is a tuple of arrays
# NUM_AGENTS = len(obs)
# print(f"Number of agents: {NUM_AGENTS}")

# # Initialize actors and critics for each agent
# actors = []
# critics = []
# optimizers = []
# for i in range(NUM_AGENTS):
#     obs_dim = obs[i].shape[0]
#     act_dim = env.action_space[i].n
#     actor = Actor(obs_dim, act_dim).to(DEVICE)
#     critic = Critic(obs_dim).to(DEVICE)
#     actors.append(actor)
#     critics.append(critic)
#     optimizers.append(optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR))

# # Training loop
# for episode in range(NUM_EPISODES):
#     obs = env.reset()
#     episode_rewards = np.zeros(NUM_AGENTS)
#     for step in range(MAX_STEPS):
#         actions = []
#         log_probs = []
#         values = []
#         obs_tensors = [torch.tensor(o, dtype=torch.float32, device=DEVICE) for o in obs]

#         # Select actions
#         for i in range(NUM_AGENTS):
#             probs = actors[i](obs_tensors[i])
#             dist = torch.distributions.Categorical(probs)
#             action = dist.sample()
#             actions.append(action.item())
#             log_probs.append(dist.log_prob(action))
#             values.append(critics[i](obs_tensors[i]))

#         # Step environment
#         next_obs, rewards, terminated, truncated, info = env.step(actions)
#         episode_rewards += np.array(rewards)

#         # Convert rewards to tensors
#         rewards_tensor = [torch.tensor(r, dtype=torch.float32, device=DEVICE) for r in rewards]
#         dones = [t or tr for t, tr in zip(terminated, truncated)]

#         # Compute losses and update networks
#         for i in range(NUM_AGENTS):
#             target = rewards_tensor[i] + (0 if dones[i] else GAMMA * critics[i](torch.tensor(next_obs[i], dtype=torch.float32, device=DEVICE)))
#             delta = target - values[i]
#             actor_loss = -log_probs[i] * delta.detach()
#             critic_loss = delta.pow(2)
#             loss = actor_loss + critic_loss

#             optimizers[i].zero_grad()
#             loss.backward()
#             optimizers[i].step()

#         obs = next_obs

#         if all(dones):
#             break

#     print(f"Episode {episode+1}, Rewards: {episode_rewards}")

# env.close()




# scripts/train_ppo_manual.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import tarware

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "tarware-tiny-3agvs-2pickers-partialobs-v1"
SAVE_DIR = "outputs/manual_ppo_models"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPISODES = 500
MAX_STEPS = 200
GAMMA = 0.99
LR = 1e-3
EPS_CLIP = 0.2

# -----------------------------
# Actor-Critic Network
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

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

# -----------------------------
# Inicializar entorno
# -----------------------------
env = gym.make(ENV_NAME)
obs_init = env.reset()  # obs_init es tupla
NUM_AGENTS = len(obs_init)
print(f"Numero de agentes: {NUM_AGENTS}")

# Crear actores, críticos y optimizadores por agente
actors, critics, optimizers = [], [], []
for i in range(NUM_AGENTS):
    obs_dim = obs_init[i].shape[0]
    act_dim = env.action_space[i].n
    actors.append(Actor(obs_dim, act_dim).to(DEVICE))
    critics.append(Critic(obs_dim).to(DEVICE))
    optimizers.append(optim.Adam(list(actors[i].parameters()) + list(critics[i].parameters()), lr=LR))

# -----------------------------
# Funciones de PPO
# -----------------------------
def select_action(actor, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    probs = actor(obs_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# -----------------------------
# Entrenamiento
# -----------------------------
for episode in range(NUM_EPISODES):
    obs = env.reset()
    episode_rewards = np.zeros(NUM_AGENTS)
    log_probs_list, values_list, rewards_list, actions_list, obs_list = [[] for _ in range(NUM_AGENTS)], [[] for _ in range(NUM_AGENTS)], [[] for _ in range(NUM_AGENTS)], [[] for _ in range(NUM_AGENTS)], [[] for _ in range(NUM_AGENTS)]

    for step in range(MAX_STEPS):
        actions = []
        log_probs = []
        values = []

        # Elegir acciones y valores
        for i in range(NUM_AGENTS):
            a, logp = select_action(actors[i], obs[i])
            v = critics[i](torch.tensor(obs[i], dtype=torch.float32, device=DEVICE))
            actions.append(a)
            log_probs.append(logp)
            values.append(v)
            obs_list[i].append(obs[i])
            actions_list[i].append(a)
            log_probs_list[i].append(logp)
            values_list[i].append(v)

        # Ejecutar paso en entorno
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += np.array(rewards)
        rewards_list = [rewards_list[i] + [rewards[i]] for i in range(NUM_AGENTS)]

        dones = [t or tr for t, tr in zip(terminated, truncated)]
        obs = next_obs

        if all(dones):
            break

    # -----------------------------
    # Actualizar redes PPO
    # -----------------------------
    for i in range(NUM_AGENTS):
        returns = []
        G = 0
        for r in reversed(rewards_list[i]):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values = torch.stack(values_list[i]).squeeze()
        log_probs = torch.stack(log_probs_list[i])

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizers[i].zero_grad()
        loss.backward()
        optimizers[i].step()

    print(f"Episode {episode+1}, Rewards: {episode_rewards}")

# -----------------------------
# Guardar modelos
# -----------------------------
for i in range(NUM_AGENTS):
    torch.save(actors[i].state_dict(), os.path.join(SAVE_DIR, f"actor_agent{i}.pt"))
    torch.save(critics[i].state_dict(), os.path.join(SAVE_DIR, f"critic_agent{i}.pt"))

print(f"Entrenamiento terminado. Modelos guardados en '{SAVE_DIR}'")
env.close()
