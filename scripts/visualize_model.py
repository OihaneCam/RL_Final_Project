# scripts/visualize_model_fixed.py
import os
import sys
import torch
import numpy as np
import gymnasium as gym
import time

# Configurar path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importar después de configurar path
from phase2_final import MultiAgentPPONetwork

def create_adapted_network(checkpoint, obs_dim, num_agents, action_dim=55):
    """Crear una red adaptada correctamente desde el checkpoint"""
    print(f"Creando red adaptada: obs_dim={obs_dim}, agents={num_agents}")
    
    valid_actions = list(range(4, 14))
    
    # Extraer config del checkpoint si existe
    config = checkpoint.get('config', {})
    
    # Crear red con las dimensiones CORRECTAS
    network = MultiAgentPPONetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        valid_actions=valid_actions,
        num_agents=num_agents,
        hidden_dim=64
    )
    
    # Obtener state dict original
    original_state_dict = checkpoint['network_state_dict']
    
    # Crear nuevo state dict adaptado
    new_state_dict = network.state_dict()
    
    # Intentar adaptar pesos capa por capa
    print("Adaptando pesos...")
    
    # Mapeo de capas que podemos transferir
    layer_mapping = {
        'encoder.2.weight': 'encoder.2.weight',
        'encoder.2.bias': 'encoder.2.bias',
        'encoder.4.weight': 'encoder.4.weight',  # Si existe en la nueva estructura
        'encoder.4.bias': 'encoder.4.bias',
    }
    
    # Para cabezas de actor/critic - solo si el número de agentes coincide
    for i in range(min(num_agents, 3)):  # Máximo 3 agentes en el modelo original
        for layer_type in ['actor', 'critic']:
            for suffix in ['.weight', '.bias']:
                old_key = f'{layer_type}_heads.{i}.2{suffix}'
                new_key = f'{layer_type}_heads.{i}.2{suffix}'
                if old_key in original_state_dict and new_key in new_state_dict:
                    if original_state_dict[old_key].shape == new_state_dict[new_key].shape:
                        new_state_dict[new_key] = original_state_dict[old_key].clone()
                        print(f"  Transferido: {old_key} -> {new_key}")
    
    # Intentar transferir encoder.0 parcialmente
    if 'encoder.0.weight' in original_state_dict and 'encoder.0.weight' in new_state_dict:
        old_weight = original_state_dict['encoder.0.weight']
        new_weight = new_state_dict['encoder.0.weight']
        
        # Las filas deben coincidir (64), las columnas difieren
        min_cols = min(old_weight.shape[1], new_weight.shape[1])
        
        if min_cols > 0:
            # Copiar las primeras min_cols columnas
            new_weight[:, :min_cols] = old_weight[:, :min_cols]
            print(f"  Transferido parcialmente encoder.0.weight: {old_weight.shape} -> {new_weight.shape}")
        
        # Bias
        if 'encoder.0.bias' in original_state_dict and 'encoder.0.bias' in new_state_dict:
            new_state_dict['encoder.0.bias'] = original_state_dict['encoder.0.bias'].clone()
            print(f"  Transferido encoder.0.bias")
    
    # Cargar el state dict adaptado
    network.load_state_dict(new_state_dict, strict=False)
    
    print("✓ Red adaptada exitosamente")
    return network

def visualize_model(model_path, env_id, num_episodes=3, render=True):
    """
    Visualiza el modelo en acción
    
    Args:
        model_path: Ruta al modelo .pt
        env_id: ID del entorno de gym
        num_episodes: Número de episodios a visualizar
        render: Mostrar animación en tiempo real
    """
    print(f"Visualizando modelo: {model_path}")
    print(f"Entorno: {env_id}")
    
    # Cargar el modelo
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Crear entorno
    try:
        base_env = gym.make(env_id)
    except Exception as e:
        print(f"Error creando entorno {env_id}: {e}")
        return
    
    num_agvs = 3 if "3agvs" in env_id else 2
    
    # Obtener dimensiones REALES del entorno
    try:
        obs_info = base_env.reset(seed=42)
        obs = obs_info[0] if isinstance(obs_info, tuple) else obs_info
        obs_dim = obs[0].shape[0] if isinstance(obs, (tuple, list)) else obs.shape[0]
        action_dim = base_env.action_space[0].n
    except Exception as e:
        print(f"Error obteniendo dimensiones: {e}")
        # Valores por defecto basados en el entorno
        obs_dim = 119 if "tiny" in env_id else 211
        action_dim = 55
        print(f"Usando valores por defecto: obs_dim={obs_dim}, action_dim={action_dim}")
    
    print(f"\nConfiguración:")
    print(f"  Agentes: {num_agvs}")
    print(f"  Observaciones: {obs_dim}")
    print(f"  Acciones: {action_dim}")
    
    # Crear red adaptada
    network = create_adapted_network(checkpoint, obs_dim, num_agvs, action_dim)
    network.eval()
    
    # Usar wrapper si está disponible
    try:
        from phase2_final import AdvancedHeuristicPickerWrapper
        env = AdvancedHeuristicPickerWrapper(base_env, num_agvs_to_control=num_agvs)
    except ImportError:
        print("  Usando entorno base (wrapper no disponible)")
        env = base_env
    
    # Ejecutar episodios
    for ep in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episodio {ep + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        obs_info = env.reset(seed=1000 + ep)
        obs = obs_info[0] if isinstance(obs_info, tuple) else obs_info
        
        # Asegurar que obs es una lista/tupla
        if not isinstance(obs, (tuple, list)):
            obs = [obs] * num_agvs
        
        total_rewards = np.zeros(num_agvs)
        total_deliveries = 0
        
        max_steps = 150  # Menos pasos para visualización rápida
        step_count = 0
        
        for step in range(max_steps):
            if render:
                try:
                    env.render()  # Mostrar visualización
                    time.sleep(0.05)  # Pequeña pausa para ver mejor
                except:
                    if step == 0:
                        print("(Render no disponible)")
                    render = False
            
            # Obtener acciones del modelo
            actions = []
            for agent_idx in range(num_agvs):
                # Asegurar que tenemos observación para este agente
                if agent_idx < len(obs):
                    agent_obs = obs[agent_idx]
                else:
                    # Si no hay observación específica, usar la primera
                    agent_obs = obs[0]
                
                obs_t = torch.tensor(
                    agent_obs.astype(np.float32),
                    dtype=torch.float32
                ).unsqueeze(0)
                
                with torch.no_grad():
                    try:
                        action, _, _ = network.get_action(
                            obs_t,
                            agent_idx=agent_idx,
                            epsilon=0.0,
                            temperature=1.0,
                            deterministic=True
                        )
                        actions.append(action.item())
                    except Exception as e:
                        print(f"Error obteniendo acción para agente {agent_idx}: {e}")
                        actions.append(0)  # Acción por defecto
            
            # Ejecutar acción
            try:
                step_result = env.step(actions)
                
                # Manejar diferentes formatos de retorno de gymnasium
                if len(step_result) == 5:
                    obs, rewards, terminated, truncated, info = step_result
                else:
                    # Formato antiguo
                    obs, rewards, terminated, info = step_result
                    truncated = terminated
                
                # Asegurar que obs es lista/tupla
                if not isinstance(obs, (tuple, list)):
                    obs = [obs] * num_agvs
                
                # Asegurar que rewards es array
                if not isinstance(rewards, (list, tuple, np.ndarray)):
                    rewards = [rewards] * num_agvs
                
                # Acumular estadísticas
                total_rewards += np.array(rewards)
                step_count += 1
                
                # Mostrar información del paso
                if info and 'shelf_deliveries' in info and info['shelf_deliveries'] > 0:
                    total_deliveries += info['shelf_deliveries']
                    print(f"  Paso {step:3d}: ✓ {info['shelf_deliveries']} entrega(s)! "
                          f"Rewards: {rewards[:2]}...")
                elif step % 30 == 0:
                    print(f"  Paso {step:3d}: Rewards: {rewards[:2]}...")
                
                # Condición de término
                if all(terminated) or all(truncated):
                    print(f"  Episodio terminado en paso {step}")
                    break
                    
            except Exception as e:
                print(f"Error en step {step}: {e}")
                break
        
        # Resumen del episodio
        print(f"\nResumen Episodio {ep + 1}:")
        print(f"  Pasos totales: {step_count}")
        print(f"  Entregas totales: {total_deliveries}")
        print(f"  Recompensa promedio: {np.mean(total_rewards):.3f}")
        
        # Pequeña pausa entre episodios
        if ep < num_episodes - 1:
            time.sleep(1)
    
    env.close()
    print(f"\n Visualización completada!")

def simple_evaluation(model_path):
    """Evaluación simple sin visualización"""
    print(f"\n{'='*60}")
    print("EVALUACIÓN SIMPLE DEL MODELO")
    print(f"{'='*60}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Información del modelo:")
    print(f"  Claves disponibles: {list(checkpoint.keys())}")
    
    if 'config' in checkpoint:
        print(f"  Configuración: {checkpoint['config'].keys()}")
    
    # Probar en diferentes entornos
    test_envs = [
        ("tarware-tiny-3agvs-2pickers-partialobs-v1", 3, 119),
        ("tarware-small-2agvs-2pickers-partialobs-v1", 2, 211),
    ]
    
    for env_id, num_agvs, expected_obs_dim in test_envs:
        print(f"\n{'='*40}")
        print(f"Probando: {env_id}")
        print(f"{'='*40}")
        
        try:
            env = gym.make(env_id)
            obs_info = env.reset(seed=42)
            obs = obs_info[0] if isinstance(obs_info, tuple) else obs_info
            
            actual_obs_dim = obs[0].shape[0] if isinstance(obs, (tuple, list)) else obs.shape[0]
            action_dim = env.action_space[0].n
            
            print(f"  Observaciones: {actual_obs_dim} (esperado: {expected_obs_dim})")
            print(f"  Acciones: {action_dim}")
            print(f"  Agentes: {num_agvs}")
            
            # Crear red adaptada
            network = create_adapted_network(checkpoint, actual_obs_dim, num_agvs, action_dim)
            
            # Probar una inferencia
            obs_t = torch.tensor(obs[0].astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits, value = network(obs_t, agent_idx=0)
                action = torch.argmax(logits).item()
                print(f"  Inferencia: acción={action}, valor={value.item():.3f}")
            
            env.close()
            
        except Exception as e:
            print(f"  Error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualizar modelo entrenado')
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al modelo .pt')
    parser.add_argument('--env', type=str, default='tarware-tiny-3agvs-2pickers-partialobs-v1',
                       help='ID del entorno de gym')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Número de episodios a visualizar')
    parser.add_argument('--no-render', action='store_true',
                       help='No mostrar animación')
    parser.add_argument('--evaluate', action='store_true',
                       help='Solo evaluación simple')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Modelo no encontrado: {args.model}")
        return
    
    if args.evaluate:
        simple_evaluation(args.model)
    else:
        visualize_model(
            model_path=args.model,
            env_id=args.env,
            num_episodes=args.episodes,
            render=not args.no_render
        )

if __name__ == "__main__":
    main()