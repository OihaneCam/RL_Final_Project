# BEST RESULTS: scripts/train_ppo_2.py
"""
FINAL Progressive Training - BUG CORREGIDO
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import tarware

# ----------------------
# FINAL Optimized Config 
# ----------------------
CONFIG = {
    # Environment
    "env_id": "tarware-tiny-3agvs-2pickers-partialobs-v1",
    
    # Training Phases
    "phases": [
        {
            "name": "agv_training",
            "episodes": 200,
            "train_agvs": True,
            "train_pickers": False,
            "use_picker_heuristic": True,
            "max_steps": 200,
            "lr": 1e-3,
            "reward_scale": 1.0,
        },
        {
            "name": "picker_training", 
            "episodes": 200,
            "train_agvs": False,
            "train_pickers": True,
            "use_picker_heuristic": False,
            "max_steps": 200,
            "lr": 1e-3,
            "reward_scale": 1.0,
        },
        {
            "name": "joint_training",
            "episodes": 300,
            "train_agvs": True,
            "train_pickers": True,
            "use_picker_heuristic": False,
            "max_steps": 300,
            "lr": 5e-4,
            "reward_scale": 1.0,
        }
    ],
    
    # Network Architecture
    "hidden_dim": 128,
    
    # PPO Parameters
    "gamma": 0.95,
    "gae_lambda": 0.90,
    "clip_eps": 0.3,
    "ppo_epochs": 2,
    "batch_size": 256,
    "max_grad_norm": 1.0,
    
    # Action Masking
    "use_action_masking": True,
    
    # Reward Engineering
    "delivery_bonus": 20.0,
    "loading_bonus": 5.0,
    "step_penalty": -0.001,
    "efficiency_bonus": 2.0,
    "exploration_bonus": 0.1,
    
    # Exploration
    "entropy_coef": 0.05,
    "min_entropy": 0.01,
    
    # Logging
    "save_interval": 25,
    "log_interval": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

DEVICE = torch.device(CONFIG["device"])

# ----------------------
# Exploratory Network
# ----------------------
class ExploratoryActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 0.5)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x, action_mask=None):
        features = self.feature_net(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        if action_mask is not None:
            logits = logits + action_mask
            
        return logits, value
    
    def get_action(self, x, action_mask=None, action=None):
        logits, value = self.forward(x, action_mask)
        
        noise = torch.randn_like(logits) * 0.1
        logits = logits + noise
        
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

# ----------------------
# Final Progressive Trainer - BUG CORREGIDO
# ----------------------
class FinalProgressiveTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        self.raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        obs0, info0 = self.safe_reset(env)
        self.n_agents = len(env.action_space)
        self.agv_ids, self.picker_ids = self.detect_agent_roles(env, info0, obs0)
        
        self.obs_dim_agv = env.observation_space[self.agv_ids[0]].shape[0]
        self.obs_dim_picker = env.observation_space[self.picker_ids[0]].shape[0]
        self.action_dim = env.action_space[0].n
        
        print(f"FINAL Progressive Training - BUG FIXED")
        print(f"   Agents: {self.n_agents} (AGVs: {self.agv_ids}, Pickers: {self.picker_ids})")
        print(f"   Device: {DEVICE}")
        
        self.policy_agv = ExploratoryActorCritic(
            self.obs_dim_agv, self.action_dim, config["hidden_dim"]
        ).to(DEVICE)
        self.policy_picker = ExploratoryActorCritic(
            self.obs_dim_picker, self.action_dim, config["hidden_dim"]
        ).to(DEVICE)
        
        self.optimizer_agv = optim.Adam(self.policy_agv.parameters(), lr=1e-3)
        self.optimizer_picker = optim.Adam(self.policy_picker.parameters(), lr=1e-3)
        
        self.current_phase = 0
        self.episode_rewards = []
        self.phase_rewards = {phase["name"]: [] for phase in config["phases"]}
        self.delivery_history = []
        
        self.total_deliveries = 0
        self.successful_episodes = 0
        self.best_reward = -float('inf')
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"models/final_fixed_{self.timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    def safe_reset(self, env, seed=None):
        out = env.reset(seed=seed) if seed is not None else env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out
            info = {}
        
        if isinstance(obs, np.ndarray) and obs.dtype == object:
            obs = list(obs)
        if not isinstance(obs, (list, tuple)):
            obs = [obs]
        
        return obs, info
    
    def detect_agent_roles(self, env, reset_info, obs_sample):
        agv_ids = []
        picker_ids = []
        
        n_agents = len(env.action_space)
        
        try:
            obs_spaces = list(env.observation_space)
            for i, space in enumerate(obs_spaces):
                if hasattr(space, 'shape') and len(space.shape) > 0:
                    if space.shape[0] >= 100:
                        agv_ids.append(i)
                    else:
                        picker_ids.append(i)
        except:
            n_agv = max(1, int(0.6 * n_agents))
            agv_ids = list(range(n_agv))
            picker_ids = list(range(n_agv, n_agents))
        
        return agv_ids, picker_ids
    
    def get_action_mask(self, agent_id):
        if not self.config["use_action_masking"]:
            return None
        
        try:
            valid_masks = self.raw_env.compute_valid_action_masks()
            mask = valid_masks[agent_id]
            logit_mask = torch.tensor(
                [0.0 if m == 1 else -1e8 for m in mask],
                device=DEVICE, dtype=torch.float32
            )
            return logit_mask
        except:
            return None
    
    def get_exploratory_heuristic(self, agent_id, step_count, agent_type):
        if agent_type == "picker":
            base = step_count % 40
            if base < 10:
                return np.random.randint(5, 15)
            elif base < 20:
                return np.random.randint(15, 30)
            elif base < 30:
                return np.random.randint(30, 45)
            else:
                return np.random.randint(1, self.action_dim)
        else:
            cycle = step_count % 50
            if cycle < 15:
                return np.random.randint(1, 8)
            elif cycle < 30:
                return np.random.randint(10, 30)
            elif cycle < 40:
                return np.random.randint(30, 45)
            else:
                return np.random.randint(1, self.action_dim)
    
    def enhance_rewards(self, rewards, info, step, max_steps):
        enhanced_rewards = []
        
        delivery_bonus = 0.0
        if info and "shelf_deliveries" in info:
            deliveries = info["shelf_deliveries"]
            delivery_bonus = deliveries * self.config["delivery_bonus"]
            self.total_deliveries += deliveries
        
        efficiency_bonus = 0.0
        if step < max_steps * 0.6:
            efficiency_bonus = self.config["efficiency_bonus"]
        
        exploration_bonus = self.config["exploration_bonus"]
        
        for i, original_reward in enumerate(rewards):
            enhanced_reward = original_reward
            
            phase_config = self.config["phases"][self.current_phase]
            enhanced_reward *= phase_config.get("reward_scale", 1.0)
            
            enhanced_reward += delivery_bonus / self.n_agents
            enhanced_reward += efficiency_bonus
            enhanced_reward += exploration_bonus
            enhanced_reward += self.config["step_penalty"]
            
            enhanced_rewards.append(enhanced_reward)
        
        return enhanced_rewards
    
    def collect_episode(self, phase_config, episode_num):
        obs, info = self.safe_reset(self.env)
        
        episode_data = {
            'observations': [[] for _ in range(self.n_agents)],
            'actions': [[] for _ in range(self.n_agents)],
            'log_probs': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'values': [[] for _ in range(self.n_agents)],
        }
        
        episode_reward = np.zeros(self.n_agents)
        episode_deliveries = 0
        
        force_exploration = episode_num < 50
        
        for step in range(phase_config["max_steps"]):
            actions = []
            action_log_probs = []
            values = []
            
            for agent_id in range(self.n_agents):
                ob = np.asarray(obs[agent_id], dtype=np.float32)
                ob_t = torch.tensor(ob, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                action_mask = self.get_action_mask(agent_id)
                
                # BUG CORREGIDO: Inicializar value aquí
                current_value = 0.0
                
                if agent_id in self.agv_ids:
                    agent_type = "agv"
                    if phase_config["train_agvs"]:
                        with torch.no_grad():
                            action, log_prob, entropy, value = self.policy_agv.get_action(
                                ob_t, action_mask
                            )
                        actions.append(action.item())
                        action_log_probs.append(log_prob.item())
                        values.append(value.item())
                        current_value = value.item()
                    else:
                        with torch.no_grad():
                            if force_exploration and np.random.random() < 0.3:
                                action = torch.tensor([np.random.randint(0, self.action_dim)])
                                current_value = 0.0  # Valor por defecto para acciones aleatorias
                            else:
                                logits, value = self.policy_agv(ob_t, action_mask)
                                probs = torch.softmax(logits, dim=-1)
                                action = torch.argmax(probs, dim=-1)
                                current_value = value.item()
                        actions.append(action.item())
                        action_log_probs.append(0.0)
                        values.append(current_value)
                
                else:
                    agent_type = "picker"
                    if phase_config["use_picker_heuristic"]:
                        action = self.get_exploratory_heuristic(agent_id, step, agent_type)
                        actions.append(action)
                        action_log_probs.append(0.0)
                        values.append(0.0)  # Valor 0 para heurística
                        current_value = 0.0
                    elif phase_config["train_pickers"]:
                        with torch.no_grad():
                            action, log_prob, entropy, value = self.policy_picker.get_action(
                                ob_t, action_mask
                            )
                        actions.append(action.item())
                        action_log_probs.append(log_prob.item())
                        values.append(value.item())
                        current_value = value.item()
                    else:
                        with torch.no_grad():
                            if force_exploration and np.random.random() < 0.3:
                                action = torch.tensor([np.random.randint(0, self.action_dim)])
                                current_value = 0.0
                            else:
                                logits, value = self.policy_picker(ob_t, action_mask)
                                probs = torch.softmax(logits, dim=-1)
                                action = torch.argmax(probs, dim=-1)
                                current_value = value.item()
                        actions.append(action.item())
                        action_log_probs.append(0.0)
                        values.append(current_value)
                
                episode_data['observations'][agent_id].append(ob)
                episode_data['actions'][agent_id].append(actions[-1])
                episode_data['log_probs'][agent_id].append(action_log_probs[-1])
                episode_data['values'][agent_id].append(current_value)  # Usar current_value
            
            next_obs, rewards, terminated, truncated, info = self.env.step(tuple(actions))
            
            if isinstance(rewards, (int, float)):
                rewards = [float(rewards)] * self.n_agents
            else:
                rewards = [float(r) for r in rewards]
            
            enhanced_rewards = self.enhance_rewards(rewards, info, step, phase_config["max_steps"])
            
            for i in range(self.n_agents):
                episode_data['rewards'][i].append(enhanced_rewards[i])
                episode_reward[i] += enhanced_rewards[i]
            
            if info and "shelf_deliveries" in info:
                episode_deliveries += info["shelf_deliveries"]
            
            obs = next_obs
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                obs = list(obs)
            if not isinstance(obs, (list, tuple)):
                obs = [obs]
            
            if (isinstance(terminated, list) and all(terminated)) or \
               (isinstance(terminated, bool) and terminated):
                break
        
        total_episode_reward = np.sum(episode_reward)
        
        if episode_deliveries > 0:
            self.successful_episodes += 1
        
        self.delivery_history.append(episode_deliveries)
        
        return episode_data, total_episode_reward, info, episode_deliveries
    
    def compute_advantages(self, rewards, values, last_value):
        returns = []
        advantages = []
        
        gae = 0
        next_value = last_value
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config["gamma"] * next_value - values[t]
            gae = delta + self.config["gamma"] * self.config["gae_lambda"] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        return returns, advantages
    
    def update_policy(self, policy, optimizer, agent_ids, episode_data, lr):
        if not agent_ids or len(episode_data['observations'][agent_ids[0]]) == 0:
            return 0.0
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        
        for agent_id in agent_ids:
            if len(episode_data['observations'][agent_id]) > 0:
                rewards = episode_data['rewards'][agent_id]
                values = episode_data['values'][agent_id]
                last_value = values[-1] if values else 0
                
                returns, advantages = self.compute_advantages(rewards, values, last_value)
                
                all_obs.extend(episode_data['observations'][agent_id])
                all_actions.extend(episode_data['actions'][agent_id])
                all_old_log_probs.extend(episode_data['log_probs'][agent_id])
                all_returns.extend(returns)
                all_advantages.extend(advantages)
        
        if len(all_obs) == 0:
            return 0.0
        
        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=DEVICE)
        old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(all_returns, dtype=torch.float32, device=DEVICE)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=DEVICE)
        
        if advantages_t.std() > 0:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        total_loss = 0
        n_updates = 0
        
        for _ in range(self.config["ppo_epochs"]):
            indices = torch.randperm(len(obs_t))
            
            for start in range(0, len(indices), self.config["batch_size"]):
                end = start + self.config["batch_size"]
                batch_indices = indices[start:end]
                
                batch_obs = obs_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                
                logits, values = policy(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                min_entropy = self.config.get("min_entropy", 0.0)
                if entropy < min_entropy:
                    entropy = torch.tensor(min_entropy)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config["clip_eps"], 1 + self.config["clip_eps"]) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, batch_returns)
                
                entropy_bonus = self.config["entropy_coef"] * entropy
                
                loss = policy_loss + 0.5 * value_loss - entropy_bonus
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config["max_grad_norm"])
                optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
        
        return total_loss / n_updates if n_updates > 0 else 0.0
    
    def run_phase(self, phase_config, phase_name):
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name.upper()} - BUG FIXED")
        print(f"{'='*60}")
        print(f"   Training AGVs: {phase_config['train_agvs']}")
        print(f"   Training Pickers: {phase_config['train_pickers']}")
        print(f"   Episodes: {phase_config['episodes']}")
        print(f"   Learning Rate: {phase_config['lr']}")
        
        phase_rewards = []
        phase_deliveries = []
        best_phase_reward = -float('inf')
        
        for episode in range(1, phase_config["episodes"] + 1):
            episode_data, total_reward, info, deliveries = self.collect_episode(phase_config, episode)
            
            agv_loss = 0.0
            picker_loss = 0.0
            
            if phase_config["train_agvs"]:
                agv_loss = self.update_policy(
                    self.policy_agv, self.optimizer_agv, 
                    self.agv_ids, episode_data, phase_config["lr"]
                )
            
            if phase_config["train_pickers"] and not phase_config["use_picker_heuristic"]:
                picker_loss = self.update_policy(
                    self.policy_picker, self.optimizer_picker,
                    self.picker_ids, episode_data, phase_config["lr"]
                )
            
            phase_rewards.append(total_reward)
            phase_deliveries.append(deliveries)
            self.episode_rewards.append(total_reward)
            self.phase_rewards[phase_name].append(total_reward)
            
            if total_reward > best_phase_reward:
                best_phase_reward = total_reward
                self.save_checkpoint(phase_name, "best")
            
            if episode % self.config["log_interval"] == 0:
                recent_rewards = phase_rewards[-self.config["log_interval"]:]
                recent_deliveries = phase_deliveries[-self.config["log_interval"]:]
                
                avg_reward = np.mean(recent_rewards)
                avg_deliveries = np.mean(recent_deliveries)
                success_rate = np.mean([1 if d > 0 else 0 for d in recent_deliveries]) * 100
                
                print(f"{phase_name} | Ep {episode:3d}/{phase_config['episodes']} | "
                      f"Reward: {total_reward:7.2f} | Avg: {avg_reward:6.2f} | "
                      f"Del: {deliveries} | AvgDel: {avg_deliveries:4.2f} | "
                      f"Success: {success_rate:5.1f}%")
            
            if episode % self.config["save_interval"] == 0:
                self.save_checkpoint(phase_name, episode)
        
        phase_avg = np.mean(phase_rewards)
        phase_std = np.std(phase_rewards)
        total_phase_deliveries = sum(phase_deliveries)
        success_rate = np.mean([1 if d > 0 else 0 for d in phase_deliveries]) * 100
        
        print(f"\n{phase_name.upper()} COMPLETED:")
        print(f"   Average Reward: {phase_avg:7.2f} ± {phase_std:6.2f}")
        print(f"   Best Reward: {best_phase_reward:7.2f}")
        print(f"   Total Deliveries: {total_phase_deliveries:4d}")
        print(f"   Success Rate: {success_rate:6.1f}%")
        print(f"   Episodes: {len(phase_rewards):3d}")
        
        return phase_rewards
    
    def train(self):
        print("Starting FINAL Progressive Training - BUG FIXED")
        start_time = time.time()
        
        for phase_idx, phase_config in enumerate(self.config["phases"]):
            phase_name = phase_config["name"]
            phase_rewards = self.run_phase(phase_config, phase_name)
            
            phase_results = {
                "phase": phase_name,
                "rewards": phase_rewards,
                "avg_reward": float(np.mean(phase_rewards)),
                "std_reward": float(np.std(phase_rewards)),
                "best_reward": float(np.max(phase_rewards)),
                "total_deliveries": self.total_deliveries,
                "successful_episodes": self.successful_episodes
            }
            
            with open(os.path.join(self.save_dir, f"{phase_name}_results.json"), "w") as f:
                json.dump(phase_results, f, indent=2)
            
            self.current_phase += 1
        
        self.save_checkpoint("final", "completed")
        
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("FINAL TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Training Time: {training_time/60:.1f} minutes")
        print(f"Total Deliveries: {self.total_deliveries}")
        print(f"Successful Episodes: {self.successful_episodes}")
        
        print(f"\nFINAL PERFORMANCE:")
        for phase_name in self.phase_rewards:
            rewards = self.phase_rewards[phase_name]
            if rewards:
                avg = np.mean(rewards)
                best = np.max(rewards)
                deliveries_in_phase = sum([1 for d in self.delivery_history[:len(rewards)] if d > 0])
                print(f"   {phase_name:15}: {avg:7.2f} (best: {best:7.2f}, deliveries: {deliveries_in_phase:3d})")
        
        self.env.close()
    
    def save_checkpoint(self, phase_name, episode_name):
        checkpoint = {
            'phase': phase_name,
            'episode': episode_name,
            'policy_agv_state_dict': self.policy_agv.state_dict(),
            'policy_picker_state_dict': self.policy_picker.state_dict(),
            'optimizer_agv_state_dict': self.optimizer_agv.state_dict(),
            'optimizer_picker_state_dict': self.optimizer_picker.state_dict(),
            'episode_rewards': self.episode_rewards,
            'phase_rewards': self.phase_rewards,
            'delivery_history': self.delivery_history,
            'total_deliveries': self.total_deliveries,
            'successful_episodes': self.successful_episodes,
            'current_phase': self.current_phase,
            'config': self.config
        }
        
        filename = f"{phase_name}_{episode_name}.pt"
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

def train():
    print("FINAL PROGRESSIVE TRAINING - BUG FIXED")
    print("="*50)
    
    env = gym.make(CONFIG["env_id"])
    trainer = FinalProgressiveTrainer(env, CONFIG)
    trainer.train()

if __name__ == "__main__":
    train()