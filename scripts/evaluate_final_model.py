# scripts/evaluate_final_model.py
"""
EVALUACIÓN COMPLETA - VERSIÓN CORREGIDA (JSON serializable)
"""
import os
import sys
import json
import numpy as np
import torch
import gymnasium as gym
import tarware
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar tipos numpy en JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class FinalHeuristicPickerWrapper(gym.Wrapper):
    """Wrapper idéntico al usado en entrenamiento"""
    def __init__(self, env):
        super().__init__(env)
        self.env_unwrapped = env.unwrapped
        self.coords_to_action_id = {v: k for k, v in env.unwrapped.action_id_to_coords_map.items()}
        
    def step(self, agv_actions):
        picker_actions = self._get_active_picker_actions(agv_actions[0])
        all_actions = list(agv_actions) + picker_actions
        obs, rewards, terminated, truncated, info = self.env.step(tuple(all_actions))
        return obs, rewards[:len(agv_actions)], terminated, truncated, info
    
    def _get_active_picker_actions(self, agv_action):
        env = self.env_unwrapped
        picker_actions = []
        
        for picker_idx in range(env.num_agvs, env.num_agents):
            picker = env.agents[picker_idx]
            action = 0
            
            agv = env.agents[0]
            target_coords = (agv.y, agv.x)
            if target_coords in self.coords_to_action_id:
                action = self.coords_to_action_id[target_coords]
                
                if (picker.x == agv.x and picker.y == agv.y and 
                    agv.req_action == 4):  # TOGGLE_LOAD
                    action = 4
            
            picker_actions.append(action)
        
        return picker_actions

class FinalAGVNetwork(torch.nn.Module):
    """Red idéntica a la usada en entrenamiento"""
    def __init__(self, obs_dim, action_dim, valid_actions):
        super().__init__()
        self.valid_actions = valid_actions
        self.action_dim = action_dim
        
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
        )
        
        self.actor = torch.nn.Linear(32, action_dim)
        self.critic = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action(self, x, epsilon=0.0, deterministic=False, 
                   temperature=1.0, forbidden_actions=None):
        logits, value = self.forward(x)
        
        mask = torch.ones_like(logits) * -1e10
        for action in self.valid_actions:
            mask[0, action] = 0
        
        if forbidden_actions:
            for action in forbidden_actions:
                mask[0, action] = -1e5
        
        masked_logits = logits + mask
        scaled_logits = masked_logits / temperature
        
        if deterministic:
            probs = torch.softmax(scaled_logits, dim=-1)
            if temperature > 0.1:
                probs = probs + 1e-6
                probs = probs / probs.sum()
                action = torch.multinomial(probs, 1).squeeze()
            else:
                action = torch.argmax(probs, dim=-1)
        else:
            if np.random.random() < epsilon:
                if forbidden_actions:
                    available_actions = [a for a in self.valid_actions 
                                       if a not in forbidden_actions]
                    if available_actions:
                        action = torch.tensor([np.random.choice(available_actions)])
                    else:
                        action = torch.tensor([np.random.choice(self.valid_actions)])
                else:
                    action = torch.tensor([np.random.choice(self.valid_actions)])
            else:
                probs = torch.softmax(scaled_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
        
        return action, torch.tensor(0.0), value

def load_model(model_path):
    """Cargar modelo y configuración"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    defaults = {
        'valid_actions': list(range(4, 14)),
        'env_id': 'tarware-tiny-1agvs-2pickers-partialobs-v1',
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return checkpoint, config

def convert_to_serializable(obj):
    """Convertir objetos numpy a tipos nativos de Python"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def evaluate_single_episode(model, env, config, episode_idx, 
                          epsilon=0.0, temperature=1.0, 
                          max_steps=150, render=False):
    """Evaluar un episodio individual"""
    obs = env.reset(seed=10000 + episode_idx)
    
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'deliveries': [],
        'positions': [],
        'action_types': []
    }
    
    total_reward = 0
    total_deliveries = 0
    step_count = 0
    
    forbidden_actions = set(config.get('forbidden_actions', []))
    
    for step in range(max_steps):
        agv = env.env_unwrapped.agents[0]
        episode_data['positions'].append((int(agv.x), int(agv.y)))  # Convertir a int
        
        obs_t = torch.tensor(
            obs[0].astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = model.get_action(
                obs_t,
                epsilon=epsilon,
                temperature=temperature,
                deterministic=(epsilon == 0.0),
                forbidden_actions=forbidden_actions
            )
        
        action_item = int(action.item())  # Convertir a int
        
        # Clasificar acción
        if action_item == 0:
            action_type = 'NOOP'
        elif 1 <= action_item <= 3:
            action_type = 'GOAL'
        elif 4 <= action_item <= 13:
            action_type = 'LOCATION'
        else:
            action_type = 'INVALID'
        
        obs, rewards, terminated, truncated, info = env.step([action_item])
        
        episode_data['observations'].append(obs[0].tolist())  # Convertir a lista
        episode_data['actions'].append(action_item)
        episode_data['rewards'].append(float(rewards[0]))  # Convertir a float
        episode_data['action_types'].append(action_type)
        
        if 'shelf_deliveries' in info and info['shelf_deliveries'] > 0:
            deliveries = int(info['shelf_deliveries'])
            total_deliveries += deliveries
            episode_data['deliveries'].append((step, deliveries))
        
        total_reward += float(rewards[0])
        step_count += 1
        
        if render:
            env.render()
        
        if all(terminated) or all(truncated):
            break
    
    unique_actions = len(set(episode_data['actions']))
    action_distribution = dict(Counter(episode_data['actions']))
    action_type_distribution = dict(Counter(episode_data['action_types']))
    
    if total_deliveries > 0:
        steps_per_delivery = step_count / total_deliveries
    else:
        steps_per_delivery = float('inf')
    
    episode_metrics = {
        'episode_idx': episode_idx,
        'total_reward': float(total_reward),
        'total_deliveries': int(total_deliveries),
        'steps': int(step_count),
        'unique_actions': int(unique_actions),
        'steps_per_delivery': float(steps_per_delivery),
        'action_distribution': action_distribution,
        'action_type_distribution': action_type_distribution,
        'success': bool(total_deliveries > 0)
    }
    
    return episode_metrics, episode_data

def evaluate_model_comprehensive(model_path, num_episodes=20, 
                               render_freq=0, save_trajectories=False):
    """Evaluación completa del modelo"""
    print("=" * 70)
    print("EVALUACIÓN COMPLETA DEL MODELO FINAL (VERSIÓN CORREGIDA)")
    print("=" * 70)
    
    print(f"\n Cargando modelo: {model_path}")
    checkpoint, config = load_model(model_path)
    
    env_id = config.get('env_id', 'tarware-tiny-1agvs-2pickers-partialobs-v1')
    print(f" Entorno: {env_id}")
    
    base_env = gym.make(env_id)
    env = FinalHeuristicPickerWrapper(base_env)
    
    obs = env.reset(seed=0)
    obs_dim = obs[0].shape[0]
    action_dim = env.action_space[0].n
    valid_actions = config.get('valid_actions', list(range(4, 14)))
    
    model = FinalAGVNetwork(obs_dim, action_dim, valid_actions)
    model.load_state_dict(checkpoint['network_state_dict'])
    model.eval()
    
    print(f"\n Configuración del modelo:")
    print(f"  Observaciones: {obs_dim} dimensiones")
    print(f"  Acciones totales: {action_dim}")
    print(f"  Acciones válidas: {len(valid_actions)} ({valid_actions})")
    print(f"  Episodio original: {checkpoint.get('episode', 'N/A')}")
    
    eval_dir = f"evaluations/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. EVALUACIÓN DEL MODELO ENTRENADO
    print(f"\n{'='*70}")
    print("1. EVALUANDO MODELO ENTRENADO")
    print(f"{'='*70}")
    
    all_episode_metrics = []
    
    for ep_idx in tqdm(range(num_episodes), desc="Evaluando episodios"):
        metrics, _ = evaluate_single_episode(
            model, env, config, ep_idx,
            epsilon=0.0,
            temperature=1.0,
            max_steps=150,
            render=(render_freq > 0 and ep_idx % render_freq == 0)
        )
        all_episode_metrics.append(metrics)
    
    # 2. EVALUACIÓN CON EXPLORACIÓN
    print(f"\n{'='*70}")
    print("2. EVALUANDO CON EXPLORACIÓN (epsilon=0.2)")
    print(f"{'='*70}")
    
    exploration_metrics = []
    for ep_idx in tqdm(range(5), desc="Evaluando con exploración"):
        metrics, _ = evaluate_single_episode(
            model, env, config, ep_idx + 1000,
            epsilon=0.2,
            temperature=1.0,
            max_steps=150,
            render=False
        )
        exploration_metrics.append(metrics)
    
    # 3. LÍNEA BASE: POLÍTICA ALEATORIA VÁLIDA
    print(f"\n{'='*70}")
    print("3. LÍNEA BASE: POLÍTICA ALEATORIA VÁLIDA")
    print(f"{'='*70}")
    
    random_metrics = []
    for ep_idx in tqdm(range(10), desc="Política aleatoria"):
        obs = env.reset(seed=20000 + ep_idx)
        total_reward = 0
        total_deliveries = 0
        actions_taken = []
        
        for step in range(150):
            action = np.random.choice(valid_actions)
            actions_taken.append(action)
            
            obs, rewards, terminated, truncated, info = env.step([action])
            
            total_reward += float(rewards[0])
            if 'shelf_deliveries' in info:
                total_deliveries += int(info['shelf_deliveries'])
            
            if all(terminated) or all(truncated):
                break
        
        random_metrics.append({
            'total_reward': float(total_reward),
            'total_deliveries': int(total_deliveries),
            'steps': step + 1,
            'unique_actions': int(len(set(actions_taken))),
            'success': bool(total_deliveries > 0)
        })
    
    # 4. ANÁLISIS ESTADÍSTICO
    print(f"\n{'='*70}")
    print("4. ANÁLISIS ESTADÍSTICO COMPLETO")
    print(f"{'='*70}")
    
    # Calcular estadísticas
    df_model = pd.DataFrame(all_episode_metrics)
    df_explore = pd.DataFrame(exploration_metrics)
    df_random = pd.DataFrame(random_metrics)
    
    # Calcular distribución de acciones combinada
    all_model_actions = []
    for metrics in all_episode_metrics:
        for action, count in metrics['action_distribution'].items():
            all_model_actions.extend([action] * count)
    
    action_counts = dict(Counter(all_model_actions))
    total_actions = len(all_model_actions)
    
    # Calcular métricas de diversidad
    if total_actions > 0:
        entropy = 0
        for count in action_counts.values():
            p = count / total_actions
            if p > 0:
                entropy -= p * np.log2(p)
        
        max_action_percentage = max(action_counts.values()) / total_actions * 100
    else:
        entropy = 0
        max_action_percentage = 0
    
    # Calcular coeficiente de Gini
    def calculate_gini(values):
        if not values:
            return 0
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        if np.sum(values) == 0:
            return 0
        return float((np.sum((2 * index - n - 1) * values)) / (n * np.sum(values)))
    
    stats = {
        'model_trained': {
            'success_rate': float(df_model['success'].mean() * 100),
            'avg_deliveries': float(df_model['total_deliveries'].mean()),
            'avg_reward': float(df_model['total_reward'].mean()),
            'avg_unique_actions': float(df_model['unique_actions'].mean()),
            'std_deliveries': float(df_model['total_deliveries'].std()),
            'max_deliveries': int(df_model['total_deliveries'].max()),
            'min_deliveries': int(df_model['total_deliveries'].min()),
            'total_successful_episodes': int(df_model['success'].sum()),
            'total_deliveries': int(df_model['total_deliveries'].sum())
        },
        'model_exploration': {
            'success_rate': float(df_explore['success'].mean() * 100),
            'avg_deliveries': float(df_explore['total_deliveries'].mean()),
            'avg_unique_actions': float(df_explore['unique_actions'].mean()),
        },
        'random_baseline': {
            'success_rate': float(df_random['success'].mean() * 100),
            'avg_deliveries': float(df_random['total_deliveries'].mean()),
            'avg_reward': float(df_random['total_reward'].mean()),
            'avg_unique_actions': float(df_random['unique_actions'].mean()),
            'total_deliveries': int(df_random['total_deliveries'].sum())
        },
        'diversity_analysis': {
            'total_unique_actions': int(len(action_counts)),
            'entropy_bits': float(entropy),
            'max_action_percentage': float(max_action_percentage),
            'gini_coefficient': float(calculate_gini(list(action_counts.values()))),
            'action_distribution': action_counts,
            'total_actions': int(total_actions)
        }
    }
    
    # 5. MOSTRAR RESULTADOS
    print(f"\n RESULTADOS DETALLADOS:")
    print(f"   Modelo entrenado:")
    print(f"     • Tasa de éxito: {stats['model_trained']['success_rate']:.1f}%")
    print(f"     • Entregas/episodio: {stats['model_trained']['avg_deliveries']:.3f}")
    print(f"     • Recompensa/episodio: {stats['model_trained']['avg_reward']:.3f}")
    print(f"     • Acciones únicas/episodio: {stats['model_trained']['avg_unique_actions']:.1f}")
    print(f"     • Entregas totales: {stats['model_trained']['total_deliveries']} en {num_episodes} episodios")
    
    print(f"\n   Línea base aleatoria:")
    print(f"     • Tasa de éxito: {stats['random_baseline']['success_rate']:.1f}%")
    print(f"     • Entregas/episodio: {stats['random_baseline']['avg_deliveries']:.3f}")
    print(f"     • Entregas totales: {stats['random_baseline']['total_deliveries']} en 10 episodios")
    
    print(f"\n   Análisis de diversidad:")
    print(f"     • Acciones únicas usadas: {stats['diversity_analysis']['total_unique_actions']}/10")
    print(f"     • Acción más usada: {stats['diversity_analysis']['max_action_percentage']:.1f}%")
    print(f"     • Entropía: {stats['diversity_analysis']['entropy_bits']:.3f} bits")
    print(f"     • Coeficiente Gini: {stats['diversity_analysis']['gini_coefficient']:.3f}")
    
    # 6. GUARDAR RESULTADOS
    print(f"\n{'='*70}")
    print("6. GUARDANDO RESULTADOS")
    print(f"{'='*70}")
    
    # Convertir todo a serializable
    results = {
        'model_path': model_path,
        'config': convert_to_serializable(config),
        'evaluation_timestamp': datetime.now().isoformat(),
        'num_evaluation_episodes': num_episodes,
        'statistics': convert_to_serializable(stats),
        'episode_details': convert_to_serializable(all_episode_metrics),
    }
    
    results_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Guardar DataFrames
    df_model.to_csv(os.path.join(eval_dir, "model_episodes.csv"), index=False)
    df_random.to_csv(os.path.join(eval_dir, "random_baseline.csv"), index=False)
    
    # 7. GENERAR GRÁFICOS
    print(f"\n{'='*70}")
    print("7. GENERANDO VISUALIZACIONES")
    print(f"{'='*70}")
    
    try:
        generate_visualizations(eval_dir, df_model, df_random, action_counts, stats)
        print(" Visualizaciones generadas con éxito")
    except Exception as e:
        print(f"⚠️  Error generando visualizaciones: {e}")
    
    # 8. RESUMEN FINAL Y RECOMENDACIONES
    print(f"\n{'='*70}")
    print(" RESUMEN FINAL")
    print(f"{'='*70}")
    
    success_rate = stats['model_trained']['success_rate']
    avg_deliveries = stats['model_trained']['avg_deliveries']
    max_action_pct = stats['diversity_analysis']['max_action_percentage']
    
    print(f"\n RENDIMIENTO:")
    print(f"  • El modelo logra una tasa de éxito del {success_rate:.1f}%")
    print(f"  • Realiza {avg_deliveries:.3f} entregas por episodio")
    print(f"  • La acción más usada representa el {max_action_pct:.1f}% del total")
    
    print(f"\n COMPARACIÓN CON BASELINE:")
    improvement = (stats['model_trained']['avg_deliveries'] - 
                  stats['random_baseline']['avg_deliveries'])
    improvement_pct = (improvement / stats['random_baseline']['avg_deliveries'] * 100 
                      if stats['random_baseline']['avg_deliveries'] > 0 else float('inf'))
    
    if improvement > 0:
        print(f"   El modelo supera el baseline en {improvement:.3f} entregas/episodio")
        print(f"     ({improvement_pct:.1f}% de mejora)")
    else:
        print(f"    El modelo NO supera el baseline")
    
    print(f"\n RECOMENDACIONES:")
    
    if success_rate > 50:
        print(f"   EXCELENTE: Tasa de éxito > 50%")
    elif success_rate > 30:
        print(f"   BUENO: Tasa de éxito > 30%")
    else:
        print(f"   MEJORABLE: Tasa de éxito < 30%")
    
    if max_action_pct < 20:
        print(f"   EXCELENTE: Buen balance de acciones (< 20% de la más usada)")
    elif max_action_pct < 30:
        print(f"   ACEPTABLE: Balance moderado (< 30% de la más usada)")
    else:
        print(f"    MEJORABLE: Acción dominante (> 30%)")
    
    if stats['diversity_analysis']['total_unique_actions'] == 10:
        print(f"   PERFECTO: Usa todas las 10 acciones válidas")
    elif stats['diversity_analysis']['total_unique_actions'] >= 7:
        print(f"   BUENO: Usa {stats['diversity_analysis']['total_unique_actions']}/10 acciones")
    else:
        print(f"    LIMITADO: Solo usa {stats['diversity_analysis']['total_unique_actions']}/10 acciones")
    
    print(f"\n Resultados guardados en: {eval_dir}")
    print(f"   • evaluation_results.json: Resultados completos")
    print(f"   • model_episodes.csv: Métricas por episodio")
    print(f"   • random_baseline.csv: Línea base")
    print(f"   • performance_analysis.png: Gráficos de análisis")
    
    env.close()
    
    return eval_dir, results

def generate_visualizations(eval_dir, df_model, df_random, action_counts, stats):
    """Generar gráficos de análisis"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Análisis del Modelo Entrenado', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Distribución de entregas
    ax1 = axes[0, 0]
    if len(df_model) > 0:
        bins = np.arange(0, df_model['total_deliveries'].max() + 2) - 0.5
        ax1.hist(df_model['total_deliveries'], bins=bins, alpha=0.7, 
                color='skyblue', edgecolor='black', label='Modelo')
        ax1.axvline(df_model['total_deliveries'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Media: {df_model["total_deliveries"].mean():.2f}')
    
    if len(df_random) > 0:
        ax1.axvline(df_random['total_deliveries'].mean(), color='green', 
                   linestyle=':', linewidth=2, label=f'Baseline: {df_random["total_deliveries"].mean():.2f}')
    
    ax1.set_xlabel('Entregas por episodio')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Entregas')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparación de éxito
    ax2 = axes[0, 1]
    categories = ['Modelo', 'Baseline']
    success_rates = [
        stats['model_trained']['success_rate'],
        stats['random_baseline']['success_rate']
    ]
    colors = ['lightgreen' if success_rates[0] > success_rates[1] else 'lightcoral', 'gray']
    bars = ax2.bar(categories, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Tasa de éxito (%)')
    ax2.set_title('Comparación de Tasa de Éxito')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Distribución de acciones
    ax3 = axes[1, 0]
    if action_counts:
        actions = sorted(action_counts.keys())
        counts = [action_counts[a] for a in actions]
        total = sum(counts)
        percentages = [c/total*100 for c in counts]
        
        colors = ['lightgreen' if a in range(4, 14) else 'lightcoral' for a in actions]
        bars = ax3.bar([str(a) for a in actions], percentages, color=colors, alpha=0.8, edgecolor='black')
        
        ax3.set_xlabel('Acción')
        ax3.set_ylabel('Porcentaje (%)')
        ax3.set_title('Distribución de Acciones')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Añadir porcentajes encima de las barras
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Gráfico 4: Recompensas por episodio
    ax4 = axes[1, 1]
    if len(df_model) > 0:
        episodes = range(1, len(df_model) + 1)
        ax4.plot(episodes, df_model['total_reward'], marker='o', linestyle='-', 
                color='blue', alpha=0.6, linewidth=1.5, markersize=4)
        ax4.axhline(y=df_model['total_reward'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Media: {df_model["total_reward"].mean():.3f}')
        
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Recompensa total')
        ax4.set_title('Recompensas por Episodio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'performance_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluación completa del modelo entrenado - Versión corregida',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Ruta al checkpoint del modelo (.pt)')
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Número de episodios de evaluación')
    parser.add_argument('--render-freq', type=int, default=0,
                       help='Renderizar cada N episodios (0 = no renderizar)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f" Error: No se encontró el archivo {args.model_path}")
        return
    
    try:
        eval_dir, results = evaluate_model_comprehensive(
            args.model_path,
            num_episodes=args.num_episodes,
            render_freq=args.render_freq
        )
        
        print(f"\n Evaluación completada exitosamente")
        print(f" Resultados en: {eval_dir}")
        
    except Exception as e:
        print(f"\n Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()