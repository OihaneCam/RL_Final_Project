import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from training.config import ENVIRONMENTS
from training.wrappers import GymV21ToGymnasiumResetWrapper, SingleAgentWrapper
from tarware.definitions import CollisionLayers


class CurriculumRewardWrapper(gym.Wrapper):
    def __init__(self, env, phase):
        super().__init__(env)
        self.phase = phase

        self.prev_picker_dist = None
        self.prev_shelf_dist = None
        self.prev_picked = 0
        self.prev_deliveries = 0
        self.no_move_steps = 0
        self.last_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_picked = 0
        self.prev_deliveries = 0
        self.no_move_steps = 0

        self.prev_picker_dist = self._dist_to_closest_picker()
        self.prev_shelf_dist = self._dist_to_shelf()
        self.last_pos = self._get_agv_pos()

        return obs, info

    def _get_agv_pos(self):
        grid = self.env.unwrapped.grid[CollisionLayers.AGVS]
        pos = np.argwhere(grid > 0)
        return tuple(pos[0]) if len(pos) > 0 else None

    def _dist_to_closest_picker(self):
        grid = self.env.unwrapped.grid
        agv_pos = self._get_agv_pos()
        if agv_pos is None:
            return 0

        pickers = np.argwhere(grid[CollisionLayers.PICKERS] > 0)
        if len(pickers) == 0:
            return 0

        return min(abs(py - agv_pos[0]) + abs(px - agv_pos[1]) for py, px in pickers)

    def _dist_to_shelf(self):
        grid = self.env.unwrapped.grid
        agv_pos = self._get_agv_pos()
        if agv_pos is None:
            return 0

        shelves = np.argwhere(grid[CollisionLayers.SHELVES] > 0)
        if len(shelves) == 0:
            return 0

        return min(abs(sy - agv_pos[0]) + abs(sx - agv_pos[1]) for sy, sx in shelves)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0


        if self.phase == 1:
            pos = self._get_agv_pos()
            if pos == self.last_pos:
                self.no_move_steps += 1
            else:
                shaped += 0.002
                self.no_move_steps = 0
            self.last_pos = pos

            if self.no_move_steps > 10:
                shaped -= 0.05

            shaped -= 0.2 * info.get("clashes", 0)

        if self.phase == 2:
            curr_dist = self._dist_to_closest_picker()
            shaped += 0.01 * (self.prev_picker_dist - curr_dist)
            self.prev_picker_dist = curr_dist

            picked = info.get("picked_items", 0)
            if picked > self.prev_picked:
                shaped += 1.0
            self.prev_picked = picked

            pos = self._get_agv_pos()
            if pos != self.last_pos:
                shaped += 0.002
                self.no_move_steps = 0
            else:
                self.no_move_steps += 1
            self.last_pos = pos

            if self.no_move_steps > 10:
                shaped -= 0.05

            shaped -= 0.2 * info.get("clashes", 0)


        if self.phase == 3:
            curr_dist = self._dist_to_closest_picker()
            shaped += 0.01 * (self.prev_picker_dist - curr_dist)
            self.prev_picker_dist = curr_dist

            picked = info.get("picked_items", 0)
            if picked > self.prev_picked:
                shaped += 1.0
            self.prev_picked = picked

            curr_shelf_dist = self._dist_to_shelf()
            shaped += 0.01 * (self.prev_shelf_dist - curr_shelf_dist)
            self.prev_shelf_dist = curr_shelf_dist

            deliveries = info.get("shelf_deliveries", 0)
            if deliveries > self.prev_deliveries:
                shaped += 5.0
            self.prev_deliveries = deliveries

            pos = self._get_agv_pos()
            if pos != self.last_pos:
                shaped += 0.002
                self.no_move_steps = 0
            else:
                self.no_move_steps += 1
            self.last_pos = pos

            if self.no_move_steps > 10:
                shaped -= 0.05

            shaped -= 0.2 * info.get("clashes", 0)

        if isinstance(reward, (list, tuple, np.ndarray)):
            reward0 = float(reward[0]) + shaped
        else:
            reward0 = float(reward) + shaped

        return obs, reward0, terminated, truncated, info


def make_env(env_name, phase):
    env = gym.make(env_name) 
    env = GymV21ToGymnasiumResetWrapper(env) 
    env = CurriculumRewardWrapper(env, phase) 
    env = SingleAgentWrapper(env) 
    return env


def train_phase(phase, timesteps, model=None):
    print(f"\n=== ENTRENANDO FASE {phase} ===")

    env_name = ENVIRONMENTS["tiny"]
    env = make_env(env_name, phase)

    model_dir = f"./models/curriculum_Prueba2_phase{phase}"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix="model"
    )

    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./tensorboard_logs/curriculum_Prueba2_phase{phase}"
        )
    else:
        model.set_env(env)

    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint,
        progress_bar=True
    )

    model.save(f"{model_dir}/final_model")
    print(f"Modelo guardado en {model_dir}/final_model")

    return model



if __name__ == "__main__":
    model = train_phase(phase=1, timesteps=300_000)
    model = train_phase(phase=2, timesteps=500_000, model=model)
    model = train_phase(phase=3, timesteps=800_000, model=model)

    print("\n=== ENTRENAMIENTO COMPLETO FINALIZADO ===")
    print("Modelo final guardado en: models/curriculum_Prueba2_phase3/final_model.zip")
