
# # Explora el entorno Tarware Tiny para entender sus espacios de acción y observación.

# import gymnasium as gym
# import tarware
# import numpy as np
# import pprint

# ENV_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"  

# def pprint_space(s):
#     try:
#         print(s)
#     except Exception:
#         print("No se pudo imprimir el space")

# def main():
#     env = gym.make(ENV_ID)
#     obs = env.reset(seed=0)
#     print("==== ENV INFO ====")
#     print("n_agents:", getattr(env, "n_agents", None))
#     print("action_space length:", len(env.action_space))
#     print("observation_space length:", len(env.observation_space))
#     print("\n-- action_space (per agent) --")
#     for i, a in enumerate(env.action_space):
#         print(f" Agent {i}: {a}")
#     print("\n-- observation_space (per agent) --")
#     for i, o in enumerate(env.observation_space):
#         print(f" Agent {i}: {o}")
#     print("\n-- sample observation (first reset) --")
#     pprint.pprint(obs)
#     # step once with random actions and print outputs
#     actions = tuple(a.sample() for a in env.action_space)
#     n_obs, reward, terminated, truncated, info = env.step(actions)
#     print("\nAfter one random step:")
#     print("actions:", actions)
#     print("rewards:", reward)
#     print("terminated:", terminated)
#     print("truncated:", truncated)
#     print("next_obs sample (agent 0):", n_obs[0] if len(n_obs)>0 else n_obs)
#     env.close()

# if __name__ == "__main__":
#     main()




# scripts/explore_tarware_env.py
"""
Explora el entorno TA-RWARE y extrae información de:
- Número de agentes
- Espacios de acción y observación
- Observaciones iniciales
- Estadísticas de las observaciones (min, max, media)
- Prueba de un paso aleatorio
- Guarda la información en JSON para posteriores experimentos
"""

import gymnasium as gym
import tarware
import numpy as np
import pprint
import json
import os

ENV_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"
OUTPUT_FILE = "env_info.json"

def pprint_space(space):
    try:
        print(space)
    except Exception:
        print("No se pudo imprimir el space")

def observation_stats(obs):
    """Calcula estadísticas básicas de cada observación por agente"""
    stats = []
    for o in obs:
        arr = np.array(o)
        stats.append({
            "shape": arr.shape,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean())
        })
    return stats

def main():
    print(f"Creando entorno {ENV_ID}")
    env = gym.make(ENV_ID)
    
    # Reset del entorno (solo devuelve obs en TA-RWARE)
    obs = env.reset(seed=0)
    
    n_agents = len(env.action_space)
    print("==== ENV INFO ====")
    print("n_agents:", n_agents)
    print("action_space length:", len(env.action_space))
    print("observation_space length:", len(env.observation_space))
    
    # Mostrar espacios de acción y observación por agente
    print("\n-- action_space (por agente) --")
    action_spaces = []
    for i, a in enumerate(env.action_space):
        pprint_space(a)
        action_spaces.append(str(a))
    
    print("\n-- observation_space (por agente) --")
    observation_spaces = []
    for i, o in enumerate(env.observation_space):
        pprint_space(o)
        observation_spaces.append(str(o))
    
    print("\n-- sample observation (primer reset) --")
    for i, o in enumerate(obs):
        print(f"Agent {i}: {o}")
    
    # Calcular estadísticas de observaciones
    obs_stats = observation_stats(obs)
    for i, s in enumerate(obs_stats):
        print(f"Agent {i} stats: {s}")
    
    # Ejecutar un paso aleatorio
    actions = tuple(a.sample() for a in env.action_space)
    n_obs, reward, terminated, truncated, info = env.step(actions)
    
    print("\n-- After one random step --")
    for i in range(n_agents):
        print(f"Agent {i}: action={actions[i]}, reward={reward[i]}, terminated={terminated[i]}, truncated={truncated[i]}")
        print(f"Next obs: {n_obs[i]}")
    
    # Guardar la info en JSON
    env_data = {
        "env_id": ENV_ID,
        "n_agents": n_agents,
        "action_spaces": action_spaces,
        "observation_spaces": observation_spaces,
        "sample_obs": [o.tolist() for o in obs],
        "sample_obs_stats": obs_stats,
        "random_step": {
            "actions": [int(a) for a in actions],
            "reward": [float(r) for r in reward],
            "terminated": [bool(t) for t in terminated],
            "truncated": [bool(t) for t in truncated],
            "next_obs": [o.tolist() for o in n_obs]
        }
    }
    
    os.makedirs("env_info", exist_ok=True)
    with open(os.path.join("env_info", OUTPUT_FILE), "w") as f:
        json.dump(env_data, f, indent=4)
    
    print(f"\nInformación del entorno guardada en env_info/{OUTPUT_FILE}")
    env.close()

if __name__ == "__main__":
    main()
