"""Configuración para experimentos."""

ENVIRONMENTS = {
    "tiny": "tarware-tiny-3agvs-2pickers-partialobs-v1",
    "small": "tarware-small-5agvs-3pickers-partialobs-v1",
}

PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
}

REWARD_CONFIG = {
    "delivery_bonus": 2.0,
    "clash_penalty": -0.5,
    "stuck_penalty": -1.0,
}