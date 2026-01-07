# scripts/phase2_final.py
"""
PHASE 2 FINAL: Entrenamiento avanzado basado en los hallazgos de Phase1
Key improvements:
1. Curriculum learning: Comenzar desde el modelo entrenado en Phase1
2. Multi-AGV training: Entrenar multiples AGVs (2-3) simultaneamente
3. Progressive complexity: Aumentar tamaño del entorno gradualmente
4. Enhanced reward shaping: Recompensas intermedias inteligentes
5. Centralized training with decentralized execution (CTDE)
6. TensorBoard integration for visualization
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
from torch.utils.tensorboard import SummaryWriter

class AdvancedHeuristicPickerWrapper(gym.Wrapper):
    """Wrapper mejorado con pickers mas inteligentes y coordinacion multi-AGV"""
    def __init__(self, env, num_agvs_to_control=3):
        super().__init__(env)
        self.env_unwrapped = env.unwrapped
        self.num_agvs_to_control = num_agvs_to_control
        self.coords_to_action_id = {v: k for k, v in env.unwrapped.action_id_to_coords_map.items()}
        
        # Mapeo inverso para encontrar coordenadas por accion
        self.action_to_coords = env.unwrapped.action_id_to_coords_map
        
        # Estado para seguimiento de misiones
        self.picker_missions = {}
        
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.picker_missions = {}
        return obs
    
    def step(self, agv_actions):
        """agv_actions es una lista con acciones para los primeros num_agvs_to_control AGVs"""
        # Asignar misiones a pickers basadas en las acciones de los AGVs
        picker_actions = self._get_intelligent_picker_actions(agv_actions)
        
        # Acciones para todos los agentes
        all_actions = list(agv_actions)
        
        # AGVs no controlados toman acciones heuristicas simples
        for i in range(self.num_agvs_to_control, self.env_unwrapped.num_agvs):
            all_actions.append(self._get_heuristic_agv_action(i))
        
        # Acciones de pickers
        all_actions += picker_actions
        
        obs, rewards, terminated, truncated, info = self.env.step(tuple(all_actions))
        
        # Extraer solo recompensas de AGVs controlados
        controlled_rewards = rewards[:self.num_agvs_to_control]
        
        return obs, controlled_rewards, terminated, truncated, info
    
    def _get_intelligent_picker_actions(self, agv_actions):
        """Picker que sigue strategicamente a los AGVs"""
        env = self.env_unwrapped
        picker_actions = []
        
        for picker_idx in range(env.num_agvs, env.num_agents):
            picker = env.agents[picker_idx]
            
            # Estrategia: ayudar al AGV mas cercano que necesita ayuda
            best_agv = None
            best_distance = float('inf')
            best_needs_help = False
            
            for agv_idx in range(self.num_agvs_to_control):
                agv = env.agents[agv_idx]
                
                # Calcular distancia Manhattan
                distance = abs(agv.x - picker.x) + abs(agv.y - picker.y)
                
                # Determinar si el AGV necesita ayuda
                needs_help = False
                if agv.carrying_shelf:
                    # Si lleva estante, necesita ayuda para descargar
                    needs_help = True
                elif agv_actions[agv_idx] in range(4, 14):
                    # Si va a un estante, podria necesitar ayuda para cargar
                    needs_help = True
                
                # Priorizar AGVs cercanos que necesiten ayuda
                if needs_help and distance < best_distance:
                    best_agv = agv
                    best_distance = distance
                    best_needs_help = needs_help
            
            if best_agv:
                # Ir a la posicion del AGV
                target_coords = (best_agv.y, best_agv.x)
                if target_coords in self.coords_to_action_id:
                    action = self.coords_to_action_id[target_coords]
                    
                    # Si ya esta en la posicion y el AGV esta listo para cargar/descargar
                    if (picker.x == best_agv.x and picker.y == best_agv.y and 
                        best_agv.req_action == 4):  # TOGGLE_LOAD
                        action = 4
                else:
                    action = 0
            else:
                # Si ningun AGV necesita ayuda inmediata, ir a posicion central
                action = 0
            
            picker_actions.append(action)
        
        return picker_actions
    
    def _get_heuristic_agv_action(self, agv_idx):
        """Accion heuristica simple para AGVs no controlados"""
        env = self.env_unwrapped
        agv = env.agents[agv_idx]
        
        if agv.carrying_shelf:
            # Si lleva estante, ir a goal mas cercano
            goals = env.goals
            if goals:
                # Goal mas cercano (simple)
                goal = goals[len(goals) // 2]  # Goal del medio
                target_coords = (goal[1], goal[0])
                if target_coords in self.coords_to_action_id:
                    return self.coords_to_action_id[target_coords]
        else:
            # Si no lleva estante, buscar estante solicitado
            for shelf in env.request_queue:
                if shelf.x != -1 and shelf.y != -1:
                    target_coords = (shelf.y, shelf.x)
                    if target_coords in self.coords_to_action_id:
                        return self.coords_to_action_id[target_coords]
        
        return 0

class MultiAgentPPONetwork(nn.Module):
    """Red para multiples AGVs con parametros compartidos pero observaciones individuales"""
    def __init__(self, obs_dim, action_dim, valid_actions, num_agents=3, hidden_dim=64):
        super().__init__()
        self.valid_actions = valid_actions
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        
        # Encoder compartido para extraer caracteristicas
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        
        # Cabezas de actor y critico separadas (una por agente)
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 32),  # Cambiado a 32 para compatibilidad
                nn.ReLU(),
                nn.Linear(32, action_dim)
            ) for _ in range(num_agents)
        ])
        
        self.critic_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 32),  # Cambiado a 32 para compatibilidad
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_agents)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        for actor_head in self.actor_heads:
            for layer in actor_head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.0)
        
        for critic_head in self.critic_heads:
            for layer in critic_head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x, agent_idx=None):
        """x: tensor de forma (batch_size, obs_dim)"""
        features = self.encoder(x)
        
        if agent_idx is not None:
            # Para un agente especifico
            logits = self.actor_heads[agent_idx](features)
            value = self.critic_heads[agent_idx](features).squeeze(-1)
        else:
            # Para todos los agentes (retorna listas)
            logits = [head(features) for head in self.actor_heads]
            values = [head(features).squeeze(-1) for head in self.critic_heads]
            return logits, values
        
        return logits, value
    
    def get_action(self, x, agent_idx, epsilon=0.0, deterministic=False, 
                   temperature=1.0, forbidden_actions=None):
        """Obtener accion para un agente especifico"""
        logits, value = self.forward(x, agent_idx)
        
        # Aplicar mascara de acciones validas
        mask = torch.ones_like(logits) * -1e8
        for action in self.valid_actions:
            mask[:, action] = 0
        
        # Penalizar acciones prohibidas
        if forbidden_actions:
            for action in forbidden_actions:
                mask[:, action] -= 1e5
        
        masked_logits = logits + mask
        
        # Temperature scaling
        scaled_logits = masked_logits / temperature
        
        if deterministic:
            # Evaluacion: softmax suavizado
            probs = torch.softmax(scaled_logits, dim=-1)
            if temperature > 0.1:
                probs = probs + 1e-6
                probs = probs / probs.sum()
                action = torch.multinomial(probs, 1).squeeze()
            else:
                action = torch.argmax(probs, dim=-1)
            
            log_prob = torch.log(probs[0, action.item()] + 1e-10)
        else:
            # Entrenamiento: epsilon-greedy con muestreo
            if np.random.random() < epsilon:
                # Exploracion: preferir acciones menos usadas
                if forbidden_actions:
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
                # Explotacion con muestreo
                probs = torch.softmax(scaled_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def adapt_to_new_env(self, new_obs_dim):
        """Adaptar la red a nuevas dimensiones de observacion"""
        if new_obs_dim != self.obs_dim:
            print(f"Adaptando red de obs_dim={self.obs_dim} a {new_obs_dim}")
            
            # Crear nuevo encoder con las dimensiones correctas
            old_encoder = self.encoder
            
            # Copiar pesos si es posible
            new_encoder = nn.Sequential(
                nn.Linear(new_obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
            )
            
            # Inicializar con los mismos pesos donde sea posible
            with torch.no_grad():
                # Capa 0: solo cambia la dimension de entrada
                if old_encoder[0].weight.shape[1] == 64 and new_encoder[0].weight.shape[1] == 64:
                    # Mismo tamaño de salida, podemos copiar parcialmente
                    min_out = min(old_encoder[0].weight.shape[0], new_encoder[0].weight.shape[0])
                    min_in = min(old_encoder[0].weight.shape[1], new_encoder[0].weight.shape[1])
                    
                    new_encoder[0].weight[:min_out, :min_in] = old_encoder[0].weight[:min_out, :min_in]
                    new_encoder[0].bias[:min_out] = old_encoder[0].bias[:min_out]
                
                # Capa 2: misma dimension de entrada/salida (64->32)
                if old_encoder[2].weight.shape == new_encoder[2].weight.shape:
                    new_encoder[2].weight.copy_(old_encoder[2].weight)
                    new_encoder[2].bias.copy_(old_encoder[2].bias)
            
            self.encoder = new_encoder
            self.obs_dim = new_obs_dim
            
            print(f"Red adaptada exitosamente")

class CurriculumManager:
    """Gestiona la complejidad progresiva del entrenamiento"""
    def __init__(self, base_env_id="tarware-tiny-1agvs-2pickers-partialobs-v1"):
        self.base_env_id = base_env_id
        self.current_stage = 0
        self.stages = [
            {"env_id": "tarware-tiny-1agvs-2pickers-partialobs-v1", 
             "num_agvs": 1, "episodes": 200, "obs_dim": 111},
            {"env_id": "tarware-tiny-2agvs-2pickers-partialobs-v1", 
             "num_agvs": 2, "episodes": 300, "obs_dim": 115},
            {"env_id": "tarware-tiny-3agvs-2pickers-partialobs-v1", 
             "num_agvs": 3, "episodes": 300, "obs_dim": 119},
            {"env_id": "tarware-small-2agvs-2pickers-partialobs-v1",
             "num_agvs": 2, "episodes": 400, "obs_dim": 115},
        ]
    
    def get_current_config(self):
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]  # Ultimo stage
    
    def advance_stage(self, performance_metric):
        """Avanzar de stage basado en metrica de desempeno"""
        threshold = 0.3  # 30% de exito para avanzar
        
        if performance_metric >= threshold and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Avanzando a STAGE {self.current_stage + 1}: {self.stages[self.current_stage]['env_id']}")
            return True
        return False

def load_phase1_model(phase1_model_path):
    """Cargar y adaptar modelo de Phase1 para Phase2"""
    print(f"Cargando modelo de Phase1: {phase1_model_path}")
    
    checkpoint = torch.load(phase1_model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    # Crear entorno para obtener dimensiones iniciales
    base_env = gym.make(config.get('env_id', 'tarware-tiny-1agvs-2pickers-partialobs-v1'))
    obs = base_env.reset(seed=0)
    obs_dim = obs[0].shape[0]
    action_dim = base_env.action_space[0].n
    valid_actions = config.get('valid_actions', list(range(4, 14)))
    
    # Crear red multi-agente
    network = MultiAgentPPONetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        valid_actions=valid_actions,
        num_agents=3,  # Maximo que usaremos
        hidden_dim=64
    )
    
    # Cargar pesos del Phase1
    phase1_state_dict = checkpoint['network_state_dict']
    new_state_dict = network.state_dict()
    
    # Mapear pesos de forma segura
    weight_mapping = {
        # Capas del encoder (las dimensiones deben coincidir)
        'shared.0.weight': 'encoder.0.weight',
        'shared.0.bias': 'encoder.0.bias',
        'shared.2.weight': 'encoder.2.weight',
        'shared.2.bias': 'encoder.2.bias',
        
        # Capas de actor (adaptar dimensiones)
        'actor.weight': 'actor_heads.0.2.weight',
        'actor.bias': 'actor_heads.0.2.bias',
        
        # Capas de critic (adaptar dimensiones)
        'critic.weight': 'critic_heads.0.2.weight',
        'critic.bias': 'critic_heads.0.2.bias',
    }
    
    for old_key, new_key in weight_mapping.items():
        if old_key in phase1_state_dict and new_key in new_state_dict:
            old_tensor = phase1_state_dict[old_key]
            new_tensor = new_state_dict[new_key]
            
            if old_tensor.shape == new_tensor.shape:
                new_state_dict[new_key].copy_(old_tensor)
                print(f"  Transferido: {old_key} -> {new_key}")
            elif 'weight' in old_key and len(old_tensor.shape) == 2:
                # Para pesos 2D, intentar copiar parcialmente
                min_dim0 = min(old_tensor.shape[0], new_tensor.shape[0])
                min_dim1 = min(old_tensor.shape[1], new_tensor.shape[1])
                
                if min_dim0 > 0 and min_dim1 > 0:
                    new_tensor[:min_dim0, :min_dim1] = old_tensor[:min_dim0, :min_dim1]
                    print(f"  Transferido parcialmente: {old_key} {old_tensor.shape} -> {new_key} {new_tensor.shape}")
            elif 'bias' in old_key and len(old_tensor.shape) == 1:
                # Para biases 1D, intentar copiar parcialmente
                min_dim = min(old_tensor.shape[0], new_tensor.shape[0])
                if min_dim > 0:
                    new_tensor[:min_dim] = old_tensor[:min_dim]
                    print(f"  Transferido parcialmente: {old_key} {old_tensor.shape} -> {new_key} {new_tensor.shape}")
            else:
                print(f"  Warning: No se pudo transferir {old_key} {old_tensor.shape} -> {new_key} {new_tensor.shape}")
    
    # Cargar el estado dict actualizado
    network.load_state_dict(new_state_dict)
    
    print(f"Modelo Phase1 cargado y adaptado para {network.num_agents} agentes")
    print(f"Parametros totales: {sum(p.numel() for p in network.parameters()):,}")
    
    return network, config

def train_phase2_curriculum(phase1_model_path, total_phases=4):
    """
    Entrenamiento Phase2 con curriculum learning y multi-agente
    """
    print("=" * 70)
    print("PHASE 2 FINAL: Entrenamiento Multi-Agente con Curriculum Learning")
    print("=" * 70)
    
    # Configuracion general
    CONFIG = {
        # Curriculum
        "total_phases": total_phases,
        
        # Hiperparametros PPO
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coef": 0.1,  # Menor entropia que Phase1 (mas explotacion)
        "clip_epsilon": 0.2,
        "ppo_epochs": 4,
        "batch_size": 64,
        
        # Exploracion
        "epsilon_start": 0.3,  # Menos exploracion inicial
        "epsilon_end": 0.05,
        "epsilon_decay": 800,
        
        # Temperature
        "temperature_start": 1.5,
        "temperature_end": 0.8,
        
        # Control de diversidad
        "max_action_percentage": 0.25,
        "forbidden_update_freq": 20,
        
        # Logging
        "log_interval": 20,
        "save_interval": 100,
        "eval_interval": 50,
        
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    
    # Semillas
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # Cargar modelo Phase1
    network, phase1_config = load_phase1_model(phase1_model_path)
    network = network.to(CONFIG["device"])
    
    # Inicializar curriculum
    curriculum = CurriculumManager()
    current_config = curriculum.get_current_config()
    
    # Directorio para Phase2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/phase2_final_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Inicializar TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    print(f"TensorBoard logs en: {os.path.join(save_dir, 'tensorboard')}")
    
    # Guardar configuracion
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump({
            "phase2_config": CONFIG,
            "phase1_original_config": phase1_config,
            "curriculum_stages": curriculum.stages
        }, f, indent=2)
    
    print(f"Directorios de modelos: {save_dir}")
    print(f"Dispositivo: {CONFIG['device']}")
    
    # Optimizador
    optimizer = optim.Adam(network.parameters(), lr=CONFIG["learning_rate"])
    
    # Estadisticas globales
    global_stats = {
        "phase_performance": [],
        "total_episodes": 0,
        "total_deliveries": 0,
        "best_performance": 0,
    }
    
    # Entrenamiento por fases
    for phase in range(CONFIG["total_phases"]):
        print(f"\n{'='*70}")
        print(f"FASE {phase + 1}: {current_config['env_id']}")
        print(f"{'='*70}")
        
        # Crear entorno para esta fase
        base_env = gym.make(current_config["env_id"])
        env = AdvancedHeuristicPickerWrapper(base_env, num_agvs_to_control=current_config["num_agvs"])
        
        # Obtener dimensiones reales del entorno
        obs = env.reset(seed=CONFIG["seed"])
        actual_obs_dim = obs[0].shape[0]
        
        print(f"  Observaciones: {actual_obs_dim} dimensiones")
        print(f"  Agentes: {current_config['num_agvs']}")
        print(f"  Episodios: {current_config['episodes']}")
        print(f"  Exploracion: epsilon={CONFIG['epsilon_start']}->{CONFIG['epsilon_end']}")
        
        # Adaptar red a nuevas dimensiones si es necesario
        if actual_obs_dim != network.obs_dim:
            print(f"  Adaptando red de {network.obs_dim} a {actual_obs_dim} dimensiones...")
            network.adapt_to_new_env(actual_obs_dim)
            network = network.to(CONFIG["device"])
            # Recrear optimizer con los nuevos parametros
            optimizer = optim.Adam(network.parameters(), lr=CONFIG["learning_rate"])
        
        # Entrenar esta fase
        phase_stats = train_single_phase(
            network=network,
            optimizer=optimizer,
            env=env,
            config=CONFIG,
            curriculum_config=current_config,
            phase_idx=phase,
            save_dir=save_dir,
            tb_writer=tb_writer,
            global_step=global_stats["total_episodes"]
        )
        
        # Actualizar estadisticas globales
        global_stats["phase_performance"].append(phase_stats)
        global_stats["total_episodes"] += phase_stats.get("total_episodes", 0)
        global_stats["total_deliveries"] += phase_stats.get("total_deliveries", 0)
        global_stats["best_performance"] = max(
            global_stats["best_performance"], 
            phase_stats.get("best_success_rate", 0)
        )
        
        # Guardar checkpoint de esta fase
        checkpoint_path = os.path.join(save_dir, f"phase_{phase+1}_checkpoint.pt")
        torch.save({
            'phase': phase + 1,
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'curriculum_config': current_config,
            'phase_stats': phase_stats,
            'global_stats': global_stats,
        }, checkpoint_path)
        
        print(f"\nResultados Fase {phase + 1}:")
        print(f"  Episodios: {phase_stats.get('total_episodes', 0)}")
        print(f"  Entregas: {phase_stats.get('total_deliveries', 0)}")
        print(f"  Tasa exito final: {phase_stats.get('final_success_rate', 0):.1f}%")
        print(f"  Mejor tasa: {phase_stats.get('best_success_rate', 0):.1f}%")
        print(f"  Acciones unicas: {phase_stats.get('avg_unique_actions', 0):.1f}")
        
        # Log a TensorBoard
        tb_writer.add_scalar(f'Phase{phase+1}/final_success_rate', 
                           phase_stats.get('final_success_rate', 0), 
                           phase + 1)
        tb_writer.add_scalar(f'Phase{phase+1}/best_success_rate', 
                           phase_stats.get('best_success_rate', 0), 
                           phase + 1)
        tb_writer.add_scalar(f'Phase{phase+1}/avg_deliveries', 
                           phase_stats.get('total_deliveries', 0) / max(1, phase_stats.get('total_episodes', 1)), 
                           phase + 1)
        
        # Evaluar si avanzar al siguiente stage del curriculum
        if phase < len(curriculum.stages) - 1:
            if curriculum.advance_stage(phase_stats.get('final_success_rate', 0) / 100):
                current_config = curriculum.get_current_config()
            else:
                print(f"No se avanza: rendimiento {phase_stats.get('final_success_rate', 0):.1f}% < 30%")
        
        env.close()
    
    # Cerrar TensorBoard writer
    tb_writer.close()
    
    # Guardar modelo final
    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'network_state_dict': network.state_dict(),
        'config': CONFIG,
        'global_stats': global_stats,
        'training_complete': True,
    }, final_model_path)
    
    # Resumen final
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO PHASE2 COMPLETADO!")
    print(f"{'='*70}")
    print(f"Resumen Final:")
    print(f"  Fases completadas: {len(global_stats['phase_performance'])}")
    print(f"  Episodios totales: {global_stats['total_episodes']}")
    print(f"  Entregas totales: {global_stats['total_deliveries']}")
    print(f"  Mejor rendimiento: {global_stats['best_performance']:.1f}%")
    print(f"  Modelo guardado en: {final_model_path}")
    print(f"  TensorBoard logs en: {os.path.join(save_dir, 'tensorboard')}")
    
    # Evaluacion final rapida
    print(f"\nEvaluacion final rapida...")
    quick_evaluate_final(network, save_dir)
    
    return save_dir

def train_single_phase(network, optimizer, env, config, curriculum_config, phase_idx, save_dir, tb_writer, global_step):
    """Entrenar una sola fase del curriculum"""
    num_episodes = curriculum_config["episodes"]
    num_agents = curriculum_config["num_agvs"]
    
    # Estado de entrenamiento
    epsilon = config["epsilon_start"]
    temperature = config["temperature_start"]
    forbidden_actions = [set() for _ in range(num_agents)]  # Por agente
    
    # Estadisticas
    phase_stats = {
        "episode_rewards": [[] for _ in range(num_agents)],
        "episode_deliveries": [],
        "success_rates": [],
        "diversity_scores": [],
        "action_distributions": [defaultdict(int) for _ in range(num_agents)],
    }
    
    best_success_rate = 0
    best_model_path = None
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Resetear entorno
        obs = env.reset(seed=config["seed"] + episode * 1000 + phase_idx * 10000)
        
        # Datos del episodio por agente
        episode_data = [[] for _ in range(num_agents)]
        episode_actions = [[] for _ in range(num_agents)]
        episode_rewards = [0] * num_agents
        episode_deliveries = 0
        
        # Ejecutar episodio
        step_count = 0
        max_steps = 200 if "small" in curriculum_config["env_id"] else 150
        
        for step in range(max_steps):
            # Obtener acciones para cada AGV controlado
            actions = []
            log_probs = []
            values = []
            
            for agent_idx in range(num_agents):
                obs_t = torch.tensor(
                    obs[agent_idx].astype(np.float32),
                    dtype=torch.float32,
                    device=config["device"]
                ).unsqueeze(0)
                
                with torch.no_grad():
                    action, log_prob, value = network.get_action(
                        obs_t, 
                        agent_idx=agent_idx,
                        epsilon=epsilon,
                        temperature=temperature,
                        forbidden_actions=forbidden_actions[agent_idx]
                    )
                
                actions.append(action.item())
                log_probs.append(log_prob.item())
                values.append(value.item())
                episode_actions[agent_idx].append(action.item())
                
                # Actualizar distribucion de acciones
                phase_stats["action_distributions"][agent_idx][action.item()] += 1
            
            # Step en el entorno
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Almacenar datos para PPO
            for agent_idx in range(num_agents):
                # Solo almacenar observaciones para el primer agente para ahorrar memoria
                obs_to_store = obs_t.cpu().numpy()[0] if agent_idx == 0 else np.zeros(network.obs_dim)
                episode_data[agent_idx].append({
                    'obs': obs_to_store,
                    'action': actions[agent_idx],
                    'log_prob': log_probs[agent_idx],
                    'value': values[agent_idx],
                    'reward': rewards[agent_idx],
                })
                episode_rewards[agent_idx] += rewards[agent_idx]
            
            # Contar entregas
            if 'shelf_deliveries' in info and info['shelf_deliveries'] > 0:
                episode_deliveries += info['shelf_deliveries']
                if episode_deliveries > 0 and episode % 50 == 0:
                    print(f"    Ep {episode}, AGVs {num_agents}: {info['shelf_deliveries']} entrega(s)")
            
            step_count += 1
            
            if all(terminated) or all(truncated):
                break
        
        # Actualizar parametros de exploracion
        decay_factor = episode / num_episodes
        epsilon = max(config["epsilon_end"],
                     config["epsilon_start"] * (1 - decay_factor))
        temperature = max(config["temperature_end"],
                         config["temperature_start"] * (1 - decay_factor))
        
        # Actualizar acciones prohibidas por agente
        if episode % config["forbidden_update_freq"] == 0:
            for agent_idx in range(num_agents):
                if episode_actions[agent_idx]:
                    recent_counts = {}
                    for action in episode_actions[agent_idx]:
                        recent_counts[action] = recent_counts.get(action, 0) + 1
                    
                    total = len(episode_actions[agent_idx])
                    
                    new_forbidden = set()
                    for action, count in recent_counts.items():
                        if count / total > config["max_action_percentage"]:
                            new_forbidden.add(action)
                    
                    forbidden_actions[agent_idx] = new_forbidden
                    if new_forbidden:
                        print(f"    Agente {agent_idx}: {len(new_forbidden)} acciones prohibidas")
        
        # Entrenar con PPO si tenemos suficientes datos
        for agent_idx in range(num_agents):
            if len(episode_data[agent_idx]) >= config["batch_size"]:
                train_multi_agent_ppo(
                    network=network,
                    optimizer=optimizer,
                    episode_data=episode_data[agent_idx],
                    config=config,
                    agent_idx=agent_idx,
                    forbidden_actions=forbidden_actions[agent_idx]
                )
        
        # Actualizar estadisticas
        for agent_idx in range(num_agents):
            phase_stats["episode_rewards"][agent_idx].append(episode_rewards[agent_idx])
        
        phase_stats["episode_deliveries"].append(episode_deliveries)
        
        success = 1 if episode_deliveries > 0 else 0
        phase_stats["success_rates"].append(success)
        
        # Calcular diversidad
        unique_actions = [len(set(actions)) for actions in episode_actions]
        avg_diversity = np.mean(unique_actions) if unique_actions else 0
        phase_stats["diversity_scores"].append(avg_diversity)
        
        # Logging a TensorBoard
        current_global_step = global_step + episode
        if tb_writer:
            # Metricas por episodio
            tb_writer.add_scalar(f'Training/Episode_reward', 
                               np.mean(episode_rewards), 
                               current_global_step)
            tb_writer.add_scalar(f'Training/Episode_deliveries', 
                               episode_deliveries, 
                               current_global_step)
            tb_writer.add_scalar(f'Training/Success', 
                               success, 
                               current_global_step)
            tb_writer.add_scalar(f'Training/Avg_diversity', 
                               avg_diversity, 
                               current_global_step)
            tb_writer.add_scalar(f'Training/Epsilon', 
                               epsilon, 
                               current_global_step)
            tb_writer.add_scalar(f'Training/Temperature', 
                               temperature, 
                               current_global_step)
            
            # Metricas por agente
            for agent_idx in range(num_agents):
                tb_writer.add_scalar(f'Agent{agent_idx}/Reward', 
                                   episode_rewards[agent_idx], 
                                   current_global_step)
                if agent_idx < len(unique_actions):
                    tb_writer.add_scalar(f'Agent{agent_idx}/Unique_actions', 
                                       unique_actions[agent_idx], 
                                       current_global_step)
        
        # Logging a consola
        if episode % config["log_interval"] == 0:
            window = min(config["log_interval"], len(phase_stats["success_rates"]))
            recent_success = phase_stats["success_rates"][-window:] if len(phase_stats["success_rates"]) >= window else phase_stats["success_rates"]
            recent_deliveries = phase_stats["episode_deliveries"][-window:] if len(phase_stats["episode_deliveries"]) >= window else phase_stats["episode_deliveries"]
            
            success_rate = np.mean(recent_success) * 100 if recent_success else 0
            avg_delivery = np.mean(recent_deliveries) if recent_deliveries else 0
            
            print(f"F{phase_idx+1} Ep {episode:4d} | "
                  f"AGVs: {num_agents} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Del: {episode_deliveries:2d} (avg: {avg_delivery:.2f}) | "
                  f"Div: {avg_diversity:.1f} | "
                  f"Epsilon: {epsilon:.2f} | "
                  f"Reward: {np.mean(episode_rewards):.3f}")
        
        # Guardar checkpoint periodico
        if config["save_interval"] > 0 and episode % config["save_interval"] == 0:
            checkpoint_path = os.path.join(save_dir, f"phase{phase_idx+1}_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'phase_stats': phase_stats,
            }, checkpoint_path)
        
        # Evaluacion periodica y guardar mejor modelo
        if config["eval_interval"] > 0 and episode % config["eval_interval"] == 0 and episode > 50:
            eval_window = min(50, len(phase_stats["success_rates"]))
            eval_success_rate = np.mean(phase_stats["success_rates"][-eval_window:]) * 100
            
            if eval_success_rate > best_success_rate:
                best_success_rate = eval_success_rate
                best_model_path = os.path.join(save_dir, f"phase{phase_idx+1}_best.pt")
                torch.save({
                    'episode': episode,
                    'network_state_dict': network.state_dict(),
                    'success_rate': best_success_rate,
                    'phase_stats': phase_stats,
                }, best_model_path)
                print(f"    Nuevo mejor modelo: {best_success_rate:.1f}% de exito")
                
                # Log a TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(f'Phase{phase_idx+1}/best_success_rate', 
                                       best_success_rate, 
                                       current_global_step)
    
    # Calcular estadisticas finales de la fase
    training_time = time.time() - start_time
    
    final_stats = {
        "total_episodes": num_episodes,
        "total_deliveries": sum(phase_stats["episode_deliveries"]),
        "final_success_rate": np.mean(phase_stats["success_rates"][-50:]) * 100 if len(phase_stats["success_rates"]) >= 50 else np.mean(phase_stats["success_rates"]) * 100,
        "best_success_rate": best_success_rate,
        "avg_unique_actions": np.mean(phase_stats["diversity_scores"]) if phase_stats["diversity_scores"] else 0,
        "training_time_minutes": training_time / 60,
        "avg_reward_per_agent": [np.mean(rewards) if rewards else 0 for rewards in phase_stats["episode_rewards"]],
        "action_distributions": [dict(dist) for dist in phase_stats["action_distributions"]],
    }
    
    print(f"\nFase completada en {training_time/60:.1f} minutos")
    
    return final_stats

def train_multi_agent_ppo(network, optimizer, episode_data, config, agent_idx, forbidden_actions):
    """PPO para un agente especifico"""
    # Preparar datos
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
    advantages, returns = compute_gae(rewards, values, config["gamma"], config["gae_lambda"])
    
    # Normalizar ventajas
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Entrenamiento PPO
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
            
            # Forward pass
            logits, batch_values = network(batch_obs, agent_idx=agent_idx)
            
            # Aplicar mascara
            mask = torch.ones_like(logits) * -1e8
            for action in network.valid_actions:
                mask[:, action] = 0
            
            # Penalizar acciones prohibidas
            if forbidden_actions:
                for action in forbidden_actions:
                    mask[:, action] -= 1e5
            
            logits = logits + mask
            
            # Distribucion
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # Perdidas
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config["clip_epsilon"], 
                              1 + config["clip_epsilon"]) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * torch.mean((batch_values - batch_returns) ** 2)
            
            entropy_bonus = config["entropy_coef"] * entropy
            
            loss = policy_loss + value_loss - entropy_bonus
            
            # Optimizar
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
            optimizer.step()

def compute_gae(rewards, values, gamma, gae_lambda):
    """Compute GAE"""
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

def quick_evaluate_final(network, save_dir):
    """Evaluacion rapida del modelo final"""
    print("\n" + "="*50)
    print("EVALUACION RAPIDA FINAL")
    print("="*50)
    
    # Configuracion de evaluacion
    eval_envs = [
        ("tarware-tiny-3agvs-2pickers-partialobs-v1", 3),
        ("tarware-small-2agvs-2pickers-partialobs-v1", 2),
    ]
    
    eval_results = {}
    
    for env_id, num_agvs in eval_envs:
        print(f"\nEvaluando: {env_id}")
        
        base_env = gym.make(env_id)
        env = AdvancedHeuristicPickerWrapper(base_env, num_agvs_to_control=num_agvs)
        
        # Adaptar red a este entorno si es necesario
        obs = env.reset(seed=0)
        actual_obs_dim = obs[0].shape[0]
        if actual_obs_dim != network.obs_dim:
            network.adapt_to_new_env(actual_obs_dim)
        
        total_deliveries = 0
        successful_episodes = 0
        
        for ep in range(5):
            obs = env.reset(seed=5000 + ep)
            ep_deliveries = 0
            
            for step in range(150):
                actions = []
                for agent_idx in range(num_agvs):
                    obs_t = torch.tensor(
                        obs[agent_idx].astype(np.float32),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        action, _, _ = network.get_action(
                            obs_t,
                            agent_idx=agent_idx,
                            epsilon=0.0,
                            temperature=1.0,
                            deterministic=True
                        )
                    actions.append(action.item())
                
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                if 'shelf_deliveries' in info:
                    ep_deliveries += info['shelf_deliveries']
                
                if all(terminated) or all(truncated):
                    break
            
            total_deliveries += ep_deliveries
            if ep_deliveries > 0:
                successful_episodes += 1
            
            print(f"  Episodio {ep+1}: {ep_deliveries} entregas")
        
        success_rate = successful_episodes / 5 * 100
        avg_deliveries = total_deliveries / 5
        
        eval_results[env_id] = {
            "success_rate": success_rate,
            "avg_deliveries": avg_deliveries,
            "total_deliveries": total_deliveries,
        }
        
        print(f"  Resultados: {success_rate:.1f}% exito, {avg_deliveries:.2f} entregas/episodio")
        
        env.close()
    
    # Guardar resultados de evaluacion
    eval_path = os.path.join(save_dir, "final_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nEvaluacion guardada en: {eval_path}")
    
    return eval_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase2: Entrenamiento avanzado multi-agente con curriculum learning'
    )
    
    parser.add_argument('--phase1-model', type=str, required=True,
                       help='Ruta al modelo entrenado en Phase1 (.pt)')
    parser.add_argument('--phases', type=int, default=4,
                       help='Numero de fases del curriculum')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.phase1_model):
        print(f"Error: No se encontro el modelo Phase1: {args.phase1_model}")
        return
    
    print("Iniciando Phase2: Mejora de resultados con curriculum learning")
    print(f"   Modelo base: {args.phase1_model}")
    print(f"   Fases: {args.phases}")
    
    save_dir = train_phase2_curriculum(
        phase1_model_path=args.phase1_model,
        total_phases=args.phases
    )
    
    print(f"\nPhase2 completado exitosamente!")
    print(f"   Resultados en: {save_dir}")
    print(f"   Para visualizar con TensorBoard: tensorboard --logdir={os.path.join(save_dir, 'tensorboard')}")

if __name__ == "__main__":
    main()