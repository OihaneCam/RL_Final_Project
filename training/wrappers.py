"""Wrappers personalizados para TA-RWARE."""

import gymnasium as gym
import numpy as np

from tarware.definitions import CollisionLayers


class GymV21ToGymnasiumResetWrapper(gym.Wrapper):
  
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        info = {}
        return obs, info


class RewardShapingWrapper(gym.Wrapper):
    
    def __init__(self, env, config):
        super().__init__(env)
        self.delivery_bonus = config["delivery_bonus"]
        self.clash_penalty = config["clash_penalty"]
        self.stuck_penalty = config["stuck_penalty"]
        self.prev_deliveries = 0
        self.prev_clashes = 0
        self.prev_stucks = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_deliveries = 0
        self.prev_clashes = 0
        self.prev_stucks = 0
        return obs, info
    
    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        curr_deliveries = info.get("shelf_deliveries", 0)
        curr_clashes = info.get("clashes", 0)
        curr_stucks = info.get("stucks", 0)
        
        new_deliveries = curr_deliveries - self.prev_deliveries
        new_clashes = curr_clashes - self.prev_clashes
        new_stucks = curr_stucks - self.prev_stucks
        
        shaped_rewards = []
        for r in rewards:
            shaped_r = r
            shaped_r += new_deliveries * self.delivery_bonus
            shaped_r += new_clashes * self.clash_penalty
            shaped_r += new_stucks * self.stuck_penalty
            shaped_rewards.append(shaped_r)
        
        self.prev_deliveries = curr_deliveries
        self.prev_clashes = curr_clashes
        self.prev_stucks = curr_stucks
        
        return obs, shaped_rewards, terminated, truncated, info


class SingleAgentWrapper(gym.Env):
        
    def __init__(self, env, picker_policy="random"):
        super().__init__()
        self.env = env
        self.num_agvs = env.unwrapped.num_agvs
        self.num_pickers = env.unwrapped.num_pickers
        self.picker_policy = picker_policy

        self._observation_space = env.observation_space[0]
        self._action_space = env.action_space[0]

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # obs es lista multi-agente: nos quedamos con AGV0
        obs0 = np.array(obs[0], dtype=np.float32).flatten()
        return obs0, info

    def step(self, action):
        # 1. Acción para AGV0 (controlado por PPO)
        actions = [int(action)]  # AGV0

        # 2. Resto de AGVs → aquí podrías poner política fija o random
        for _ in range(self.num_agvs - 1):
            actions.append(self.env.action_space[0].sample())

        # 3. Pickers → política aleatoria
        for i in range(self.num_pickers):
            actions.append(self.env.action_space[self.num_agvs + i].sample())

        # 4. Ejecutar en el entorno multi‑agente
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        # 5. Convertir outputs a formato single‑agent (AGV0)
        obs0 = np.array(obs[0], dtype=np.float32).flatten()

        # Asegurar tipos correctos por si vienen como listas/arrays
        if isinstance(rewards, (list, tuple, np.ndarray)):
            r0 = float(rewards[0])
        else:
            r0 = float(rewards)

        if isinstance(terminated, (list, tuple, np.ndarray)):
            term0 = bool(terminated[0])
        else:
            term0 = bool(terminated)

        if isinstance(truncated, (list, tuple, np.ndarray)):
            trunc0 = bool(truncated[0])
        else:
            trunc0 = bool(truncated)

        return obs0, r0, term0, trunc0, info
