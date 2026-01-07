# scripts/phase1_final.py
"""
FASE 1 FINAL: Entrenamiento que garantiza diversidad en evaluación
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import tarware
import time
from datetime import datetime
from collections import deque, defaultdict

class FinalHeuristicPickerWrapper(gym.Wrapper):
    """Wrapper final optimizado"""
    def __init__(self, env):
        super().__init__(env)
        self.env_unwrapped = env.unwrapped
        self.coords_to_action_id = {v: k for k, v in env.unwrapped.action_id_to_coords_map.items()}
        
    def step(self, agv_actions):
        # Pickers ayudan activamente
        picker_actions = self._get_active_picker_actions(agv_actions[0])
        
        all_actions = list(agv_actions) + picker_actions
        obs, rewards, terminated, truncated, info = self.env.step(tuple(all_actions))
        
        return obs, rewards[:len(agv_actions)], terminated, truncated, info
    
    def _get_active_picker_actions(self, agv_action):
        """Pickers que activamente ayudan al AGV"""
        env = self.env_unwrapped
        picker_actions = []
        
        for picker_idx in range(env.num_agvs, env.num_agents):
            picker = env.agents[picker_idx]
            action = 0
            
            # Estrategia simple: seguir al AGV principal
            agv = env.agents[0]
            
            # Ir a la posición del AGV
            target_coords = (agv.y, agv.x)
            if target_coords in self.coords_to_action_id:
                action = self.coords_to_action_id[target_coords]
                
                # Si ya está en la posición y el AGV está listo, ayudar
                if (picker.x == agv.x and picker.y == agv.y and 
                    agv.req_action == 4):  # TOGGLE_LOAD
                    action = 4
            
            picker_actions.append(action)
        
        return picker_actions

class FinalAGVNetwork(nn.Module):
    """Red final simple pero efectiva"""
    def __init__(self, obs_dim, action_dim, valid_actions):
        super().__init__()
        self.valid_actions = valid_actions
        self.action_dim = action_dim
        
        # Arquitectura simple para evitar sobreparametrización
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        
        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        # Inicialización uniforme para diversidad
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.1, 0.1)
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.uniform_(self.actor.weight, -0.01, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        
        nn.init.uniform_(self.critic.weight, -0.1, 0.1)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action(self, x, epsilon=0.0, deterministic=False, 
                   temperature=1.0, forbidden_actions=None):
        """Obtener acción con penalización de acciones prohibidas"""
        logits, value = self.forward(x)
        
        # Aplicar máscara de acciones válidas
        mask = torch.ones_like(logits) * -1e10  # Muy negativo
        for action in self.valid_actions:
            mask[0, action] = 0
        
        # Penalizar acciones prohibidas (usadas en exceso)
        if forbidden_actions:
            for action in forbidden_actions:
                mask[0, action] = -1e5  # Penalización muy fuerte
        
        masked_logits = logits + mask
        
        # Temperature scaling
        scaled_logits = masked_logits / temperature
        
        if deterministic:
            # En evaluación: usar softmax con temperatura para suavizar
            probs = torch.softmax(scaled_logits, dim=-1)
            # Añadir ruido pequeño para evitar determinismo total
            if temperature > 0.1:
                probs = probs + 1e-6
                probs = probs / probs.sum()
                action = torch.multinomial(probs, 1).squeeze()
            else:
                action = torch.argmax(probs, dim=-1)
            
            log_prob = torch.log(probs[0, action.item()] + 1e-10)
        else:
            # En entrenamiento: epsilon-greedy mejorado
            if np.random.random() < epsilon:
                # Exploración: preferir acciones menos usadas
                if forbidden_actions:
                    # Filtrar acciones no prohibidas
                    available_actions = [a for a in self.valid_actions 
                                       if a not in forbidden_actions]
                    if available_actions:
                        action = torch.tensor([np.random.choice(available_actions)], 
                                            device=x.device)
                    else:
                        action = torch.tensor([np.random.choice(self.valid_actions)], 
                                            device=x.device)
                else:
                    action = torch.tensor([np.random.choice(self.valid_actions)], 
                                        device=x.device)
                
                probs = torch.softmax(scaled_logits, dim=-1)
                log_prob = torch.log(probs[0, action.item()] + 1e-10)
            else:
                # Explotación con muestreo suavizado
                probs = torch.softmax(scaled_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action, log_prob, value

def train_final_phase1():
    print("=" * 60)
    print("FASE 1 FINAL: Entrenamiento con diversidad garantizada")
    print("=" * 60)
    
    CONFIG = {
        "env_id": "tarware-tiny-1agvs-2pickers-partialobs-v1",
        "total_episodes": 600,  # Menos episodios pero mejor calidad
        "max_steps": 150,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coef": 0.2,  # Mucha entropía para diversidad
        "clip_epsilon": 0.3,  # Más flexible
        "ppo_epochs": 3,
        "batch_size": 32,
        "valid_actions": list(range(4, 14)),
        
        # Exploración agresiva
        "epsilon_start": 0.9,
        "epsilon_end": 0.3,  # Mantener exploración alta
        "epsilon_decay": 500,
        
        # Temperature para suavizar decisiones
        "temperature_start": 3.0,
        "temperature_end": 1.0,  # Mantener alta para diversidad
        
        # Control de acciones dominantes
        "max_action_percentage": 0.3,  # Máximo 30% para cualquier acción
        "forbidden_update_freq": 10,  # Actualizar cada 10 episodios
        
        "log_interval": 10,
        "save_interval": 50,
        "device": "cpu",
        "seed": 42,
    }
    
    # Semillas
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # Entorno
    print(f"Entorno: {CONFIG['env_id']}")
    base_env = gym.make(CONFIG["env_id"])
    env = FinalHeuristicPickerWrapper(base_env)
    
    obs = env.reset(seed=CONFIG["seed"])
    obs_dim = obs[0].shape[0]
    action_dim = env.action_space[0].n
    
    print(f"\nDimensiones:")
    print(f"  Observacion: {obs_dim}")
    print(f"  Acciones: {action_dim}")
    print(f"  Validas: {len(CONFIG['valid_actions'])}")
    
    # Red
    network = FinalAGVNetwork(obs_dim, action_dim, CONFIG["valid_actions"]).to(CONFIG["device"])
    optimizer = optim.Adam(network.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"\nRed creada: {sum(p.numel() for p in network.parameters()):,} parametros")
    
    # Directorio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/phase1_final_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\nIniciando entrenamiento...")
    print(f"Modelos en: {save_dir}")
    
    # Estadísticas
    action_history = deque(maxlen=200)  # Historial más largo
    episode_rewards = []
    episode_deliveries = []
    diversity_scores = []
    
    epsilon = CONFIG["epsilon_start"]
    temperature = CONFIG["temperature_start"]
    
    start_time = time.time()
    best_balanced_score = 0
    forbidden_actions = set()  # Acciones prohibidas por uso excesivo
    
    for episode in range(CONFIG["total_episodes"]):
        obs = env.reset(seed=CONFIG["seed"] + episode * 1000)
        episode_data = []
        episode_actions = []
        episode_reward = 0
        episode_delivery = 0
        
        for step in range(CONFIG["max_steps"]):
            obs_t = torch.tensor(
                obs[0].astype(np.float32),
                dtype=torch.float32,
                device=CONFIG["device"]
            ).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = network.get_action(
                    obs_t, 
                    epsilon=epsilon,
                    temperature=temperature,
                    forbidden_actions=forbidden_actions
                )
            
            action_item = action.item()
            episode_actions.append(action_item)
            action_history.append(action_item)
            
            obs, rewards, terminated, truncated, info = env.step([action_item])
            
            episode_data.append({
                'obs': obs_t.cpu().numpy()[0],
                'action': action_item,
                'log_prob': log_prob.item(),
                'value': value.item(),
                'reward': rewards[0],
            })
            
            episode_reward += rewards[0]
            
            if 'shelf_deliveries' in info and info['shelf_deliveries'] > 0:
                episode_delivery += info['shelf_deliveries']
                print(f"    Ep {episode}, Paso {step}: {info['shelf_deliveries']} entrega(s)")
            
            if all(terminated) or all(truncated):
                break
        
        # Actualizar parámetros
        epsilon = max(CONFIG["epsilon_end"],
                     CONFIG["epsilon_start"] * (1 - episode / CONFIG["epsilon_decay"]))
        
        temperature = max(CONFIG["temperature_end"],
                         CONFIG["temperature_start"] * (1 - episode / CONFIG["epsilon_decay"]))
        
        # Actualizar acciones prohibidas
        if episode % CONFIG["forbidden_update_freq"] == 0 and len(action_history) > 0:
            # Calcular distribución reciente
            recent_counts = defaultdict(int)
            for a in action_history:
                recent_counts[a] += 1
            
            total_recent = len(action_history)
            new_forbidden = set()
            
            for action, count in recent_counts.items():
                percentage = count / total_recent
                if percentage > CONFIG["max_action_percentage"]:
                    new_forbidden.add(action)
                    print(f"    Accion {action} prohibida: {percentage:.1%} > {CONFIG['max_action_percentage']:.0%}")
            
            forbidden_actions = new_forbidden
        
        # Calcular diversidad
        unique_actions = len(set(episode_actions))
        diversity_scores.append(unique_actions)
        
        # Entrenar si tenemos suficientes datos
        if len(episode_data) >= CONFIG["batch_size"]:
            train_final_ppo(network, optimizer, episode_data, CONFIG, forbidden_actions)
        
        # Guardar estadísticas
        episode_rewards.append(episode_reward)
        episode_deliveries.append(episode_delivery)
        
        # Calcular score balanceado (diversidad + entregas)
        balanced_score = unique_actions * (1 + episode_delivery * 0.5)
        
        # Guardar mejor modelo balanceado
        if balanced_score > best_balanced_score and episode > 20:
            best_balanced_score = balanced_score
            
            # Guardar modelo
            torch.save({
                'episode': episode,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'diversity': unique_actions,
                'deliveries': episode_delivery,
                'balanced_score': balanced_score,
                'forbidden_actions': list(forbidden_actions),
            }, os.path.join(save_dir, "best_model_final.pt"))
            
            print(f"    Mejor modelo: {episode_delivery} entregas, {unique_actions} acciones, score: {balanced_score:.1f}")
        
        # Logging
        if episode % CONFIG["log_interval"] == 0:
            # Estadísticas de ventana
            window = min(CONFIG["log_interval"], len(episode_deliveries))
            recent_del = episode_deliveries[-window:] if len(episode_deliveries) >= window else episode_deliveries
            recent_div = diversity_scores[-window:] if len(diversity_scores) >= window else diversity_scores
            
            avg_delivery = np.mean(recent_del) if recent_del else 0
            avg_diversity = np.mean(recent_div) if recent_div else 0
            success_rate = np.mean([1 if d > 0 else 0 for d in recent_del]) * 100 if recent_del else 0
            
            # Distribución de este episodio
            ep_counts = defaultdict(int)
            for a in episode_actions:
                ep_counts[a] += 1
            
            # Top 3 acciones de este episodio
            top_actions = sorted(ep_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            action_str = ", ".join([f"{a}({c/len(episode_actions)*100:.0f}%)" for a, c in top_actions])
            
            # Info de acciones prohibidas
            forbidden_str = f"Forb: {len(forbidden_actions)}" if forbidden_actions else "Forb: 0"
            
            print(f"Ep {episode:4d} | "
                  f"Del: {episode_delivery:2d} (avg: {avg_delivery:.2f}) | "
                  f"Div: {unique_actions:2d} (avg: {avg_diversity:.1f}) | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Reward: {episode_reward:6.3f} | "
                  f"Eps: {epsilon:.2f}, T: {temperature:.1f} | "
                  f"Top: {action_str} | {forbidden_str}")
        
        # Checkpoint
        if CONFIG["save_interval"] > 0 and episode % CONFIG["save_interval"] == 0:
            torch.save({
                'episode': episode,
                'network_state_dict': network.state_dict(),
                'config': CONFIG,
            }, os.path.join(save_dir, f"checkpoint_ep{episode}.pt"))
    
    # Guardar modelo final
    torch.save({
        'episode': CONFIG["total_episodes"],
        'network_state_dict': network.state_dict(),
        'config': CONFIG,
        'final_stats': {
            'avg_deliveries': np.mean(episode_deliveries) if episode_deliveries else 0,
            'avg_diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'success_rate': np.mean([1 if d > 0 else 0 for d in episode_deliveries]) * 100 if episode_deliveries else 0,
        }
    }, os.path.join(save_dir, "final_model.pt"))
    
    # Estadísticas finales
    training_time = time.time() - start_time
    
    # Distribución final de acciones
    final_counts = defaultdict(int)
    for a in action_history:
        final_counts[a] += 1
    
    stats = {
        'total_episodes': len(episode_rewards),
        'total_deliveries': sum(episode_deliveries),
        'success_rate': np.mean([1 if d > 0 else 0 for d in episode_deliveries]) * 100,
        'avg_deliveries': np.mean(episode_deliveries),
        'avg_diversity': np.mean(diversity_scores),
        'max_diversity': max(diversity_scores),
        'best_balanced_score': best_balanced_score,
        'action_distribution': dict(final_counts),
        'training_time_minutes': training_time / 60,
    }
    
    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("ENTRENAMIENTO FINAL COMPLETADO!")
    print("=" * 60)
    print(f"Tiempo: {training_time/60:.1f} minutos")
    print(f"Resultados:")
    print(f"  Episodios: {stats['total_episodes']}")
    print(f"  Entregas totales: {stats['total_deliveries']}")
    print(f"  Tasa de exito: {stats['success_rate']:.1f}%")
    print(f"  Entregas/episodio: {stats['avg_deliveries']:.2f}")
    print(f"  Diversidad promedio: {stats['avg_diversity']:.1f} acciones unicas")
    
    # Distribución detallada
    print(f"\nDistribucion de acciones (historial completo):")
    total_actions = len(action_history)
    for action in sorted(final_counts.keys()):
        count = final_counts[action]
        percentage = (count / total_actions) * 100
        is_valid = "V" if action in CONFIG['valid_actions'] else "I"
        is_forbidden = "F" if action in forbidden_actions else " "
        print(f"  {is_valid}{is_forbidden} Accion {action:3d}: {count:5d} veces ({percentage:5.1f}%)")
    
    # Evaluación INMEDIATA integrada
    print(f"\n" + "=" * 60)
    print("EVALUACION INMEDIATA DETERMINISTA")
    print("=" * 60)
    
    # Cargar mejor modelo y evaluar DIRECTAMENTE
    try:
        # Usar la MISMA red para evaluación
        network.eval()
        
        # Estadísticas de evaluación
        eval_deliveries = 0
        eval_actions = []
        
        for ep in range(5):
            obs = env.reset(seed=8000 + ep)
            ep_actions = []
            ep_deliveries = 0
            
            for step in range(150):
                obs_t = torch.tensor(obs[0].astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    # Usar temperatura 1.0 para suavizar pero ser determinista
                    action, _, _ = network.get_action(
                        obs_t, 
                        epsilon=0.0,  # Sin exploración
                        temperature=1.0,  # Temperatura media
                        deterministic=True,
                        forbidden_actions=forbidden_actions
                    )
                
                action_item = action.item()
                ep_actions.append(action_item)
                eval_actions.append(action_item)
                
                obs, rewards, terminated, truncated, info = env.step([action_item])
                
                if 'shelf_deliveries' in info:
                    ep_deliveries += info['shelf_deliveries']
                
                if all(terminated) or all(truncated):
                    break
            
            eval_deliveries += ep_deliveries
            print(f"Episodio {ep+1}: {ep_deliveries} entregas, {len(set(ep_actions))} acciones unicas")
        
        # Análisis de evaluación
        print(f"\nResultados evaluacion determinista:")
        print(f"  Entregas totales: {eval_deliveries} en 5 episodios")
        print(f"  Promedio: {eval_deliveries/5:.2f} entregas/episodio")
        
        if eval_actions:
            eval_counts = defaultdict(int)
            for a in eval_actions:
                eval_counts[a] += 1
            
            total_eval = len(eval_actions)
            unique_eval = len(set(eval_actions))
            
            print(f"\n  Distribucion en evaluacion:")
            for action in sorted(eval_counts.keys()):
                count = eval_counts[action]
                percentage = (count / total_eval) * 100
                is_valid = "V" if action in CONFIG['valid_actions'] else "I"
                print(f"    {is_valid} Accion {action:3d}: {count:4d} veces ({percentage:5.1f}%)")
            
            print(f"\n  Diversidad en evaluacion: {unique_eval} acciones unicas")
            
            # Verificar balance
            max_percentage = max(eval_counts.values()) / total_eval * 100 if total_eval > 0 else 0
            print(f"  Accion mas usada: {max_percentage:.1f}%")
            
            if max_percentage > 50:
                print(f"  ADVERTENCIA: Accion dominante (>50%) en evaluacion")
            elif max_percentage > 30:
                print(f"  AVISO: Accion bastante usada (>30%) en evaluacion")
            else:
                print(f"  EXCELENTE: Buen balance en evaluacion")
            
            if eval_deliveries > 0:
                print(f"\n  EXITO: El modelo hace entregas en evaluacion determinista!")
            else:
                print(f"\n  FALLO: El modelo NO hace entregas en evaluacion determinista")
    
    except Exception as e:
        print(f"Error en evaluacion: {e}")
    
    return save_dir

def train_final_ppo(network, optimizer, episode_data, config, forbidden_actions):
    """PPO final con control de diversidad"""
    # Convertir a tensores eficientemente
    obs_array = np.array([step['obs'] for step in episode_data])
    obs = torch.tensor(obs_array, dtype=torch.float32, device=config["device"])
    
    actions = torch.tensor(
        [step['action'] for step in episode_data],
        dtype=torch.long,
        device=config["device"]
    )
    old_log_probs = torch.tensor(
        [step['log_prob'] for step in episode_data],
        dtype=torch.float32,
        device=config["device"]
    )
    rewards = torch.tensor(
        [step['reward'] for step in episode_data],
        dtype=torch.float32,
        device=config["device"]
    )
    values = torch.tensor(
        [step['value'] for step in episode_data],
        dtype=torch.float32,
        device=config["device"]
    )
    
    # Calcular GAE
    advantages, returns = compute_final_gae(rewards, values, config["gamma"], config["gae_lambda"])
    
    # Normalizar ventajas
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Múltiples épocas PPO
    for epoch in range(config["ppo_epochs"]):
        # Mezclar
        indices = torch.randperm(len(obs))
        
        for start in range(0, len(indices), config["batch_size"]):
            end = start + config["batch_size"]
            batch_idx = indices[start:end]
            
            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            
            # Forward
            logits, batch_values = network(batch_obs)
            
            # Aplicar máscara con penalizaciones
            mask = torch.ones_like(logits) * -1e8
            for action in network.valid_actions:
                mask[:, action] = 0
            
            # Penalizar acciones prohibidas
            if forbidden_actions:
                for action in forbidden_actions:
                    mask[:, action] -= 1e5  # Penalización fuerte
            
            logits = logits + mask
            
            # Distribución
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # Pérdidas
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config["clip_epsilon"], 
                              1 + config["clip_epsilon"]) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * torch.mean((batch_values - batch_returns) ** 2)
            
            # Bonus de entropía alto para diversidad
            entropy_bonus = config["entropy_coef"] * entropy
            
            # Pérdida total
            loss = policy_loss + value_loss - entropy_bonus
            
            # Optimizar
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping suave
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            
            optimizer.step()

def compute_final_gae(rewards, values, gamma, gae_lambda):
    """GAE para entrenamiento final"""
    advantages = []
    gae = 0
    
    next_values = torch.cat([values[1:], torch.tensor([0.0], device=values.device)])
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, device=rewards.device)
    returns = advantages + values
    
    return advantages, returns

if __name__ == "__main__":
    save_dir = train_final_phase1()