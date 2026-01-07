import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from training.config import ENVIRONMENTS
from training.wrappers import GymV21ToGymnasiumResetWrapper
from training.train_curriculum import CurriculumRewardWrapper


def make_render_env(env_name):
    env = gym.make(env_name)
    env = GymV21ToGymnasiumResetWrapper(env)
    env = CurriculumRewardWrapper(env, phase=3)
    return env

def evaluate_with_render(model, env, episodes=3):
    unwrapped = env.unwrapped
    num_agvs = unwrapped.num_agvs
    num_pickers = unwrapped.num_pickers

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        total_deliveries = 0
        total_clashes = 0
        total_stucks = 0
        total_agv_distance = 0
        total_picker_distance = 0
        total_agv_idle = 0
        total_picker_idle = 0

        print(f"\n=== Episodio {ep+1} ===")

        while not done:
            obs0 = np.array(obs[0], dtype=np.float32).flatten()
            action0, _ = model.predict(obs0, deterministic=True)
            action0 = int(action0)

            actions = [action0] * num_agvs

            for i in range(num_pickers):
                actions.append(env.action_space[num_agvs + i].sample())

            obs, rewards, terminated, truncated, info = env.step(actions)

            env.render()

            total_deliveries += info.get("shelf_deliveries", 0)
            total_clashes += info.get("clashes", 0)
            total_stucks += info.get("stucks", 0)
            total_agv_distance += info.get("agvs_distance_travelled", 0)
            total_picker_distance += info.get("pickers_distance_travelled", 0)
            total_agv_idle += info.get("agvs_idle_time", 0)
            total_picker_idle += info.get("pickers_idle_time", 0)

            # reward
            if isinstance(rewards, (list, tuple, np.ndarray)):
                total_reward += float(np.mean(rewards[:num_agvs]))
            else:
                total_reward += float(rewards)

            if isinstance(terminated, (list, tuple, np.ndarray)):
                done = bool(any(terminated)) or bool(any(truncated))
            else:
                done = bool(terminated) or bool(truncated)

        print(f"Episode reward: {total_reward}")
        print("Episode info:")
        print(f"  Total deliveries: {total_deliveries}")
        print(f"  Total clashes: {total_clashes}")
        print(f"  Total stucks: {total_stucks}")
        print(f"  AGV distance travelled: {total_agv_distance}")
        print(f"  Picker distance travelled: {total_picker_distance}")
        print(f"  AGV idle time: {total_agv_idle}")
        print(f"  Picker idle time: {total_picker_idle}")


if __name__ == "__main__":
    env_name = ENVIRONMENTS["tiny"]
    # model_path = "./models/curriculum_phase3/final_model.zip"
    # model_path = "./models/curriculum_Prueba2_phase3/model_600000_steps.zip"
    model_path = "./models/curriculum_v2/phase3/final_model.zip"


    model = PPO.load(model_path)
    env = make_render_env(env_name)

    evaluate_with_render(model, env, episodes=3)
