
# Explora el entorno Tarware Tiny para entender sus espacios de acción y observación.

import gymnasium as gym
import tarware
import numpy as np
import pprint

ENV_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"  # cambia si hace falta

def pprint_space(s):
    try:
        print(s)
    except Exception:
        print("No se pudo imprimir el space")

def main():
    env = gym.make(ENV_ID)
    obs = env.reset(seed=0)
    print("==== ENV INFO ====")
    print("n_agents:", getattr(env, "n_agents", None))
    print("action_space length:", len(env.action_space))
    print("observation_space length:", len(env.observation_space))
    print("\n-- action_space (per agent) --")
    for i, a in enumerate(env.action_space):
        print(f" Agent {i}: {a}")
    print("\n-- observation_space (per agent) --")
    for i, o in enumerate(env.observation_space):
        print(f" Agent {i}: {o}")
    print("\n-- sample observation (first reset) --")
    pprint.pprint(obs)
    # step once with random actions and print outputs
    actions = tuple(a.sample() for a in env.action_space)
    n_obs, reward, terminated, truncated, info = env.step(actions)
    print("\nAfter one random step:")
    print("actions:", actions)
    print("rewards:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("next_obs sample (agent 0):", n_obs[0] if len(n_obs)>0 else n_obs)
    env.close()

if __name__ == "__main__":
    main()

