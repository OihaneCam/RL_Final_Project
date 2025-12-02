# scripts/evaluate_model_2_with_bonus.py
# python -m scripts.evaluate_model_2_bonusRewards --model-path models/final_fixed_20251127_112813/final_completed.pt --num-episodes 3
"""
Evaluator that matches training reward conditions
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import gymnasium as gym
import tarware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_ppo_2 import ExploratoryActorCritic

class TrainingConditionEvaluator:
    def __init__(self, model_path, env_id=None, render=True, num_episodes=5, max_steps=500):
        self.model_path = model_path
        self.render = render
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Load checkpoint
        print(f"📂 Loading model from: {model_path}")
        try:
            self.checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except:
            self.checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.config = self.checkpoint.get('config', {})
        
        # Use provided env_id or from config
        self.env_id = env_id or self.config.get('env_id', 'tarware-tiny-3agvs-2pickers-partialobs-v1')
        
        # Create environment
        print(f"🎯 Creating environment: {self.env_id}")
        self.env = gym.make(self.env_id)
        
        # Get basic environment info
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs0, info0 = reset_result
        else:
            obs0 = reset_result
            info0 = {}
            
        self.n_agents = len(self.env.action_space)
        
        # Detect agent roles
        self.agv_ids, self.picker_ids = self.detect_agent_roles(self.env, info0, obs0)
        
        # Get observation and action dimensions
        self.obs_dim_agv = self.env.observation_space[self.agv_ids[0]].shape[0]
        self.obs_dim_picker = self.env.observation_space[self.picker_ids[0]].shape[0]
        self.action_dim = self.env.action_space[0].n
        
        # Create policies
        self.policy_agv = ExploratoryActorCritic(self.obs_dim_agv, self.action_dim, 
                                                self.config.get("hidden_dim", 128))
        self.policy_picker = ExploratoryActorCritic(self.obs_dim_picker, self.action_dim, 
                                                   self.config.get("hidden_dim", 128))
        
        # Load trained weights
        self._load_model_weights()
        
        self.policy_agv.eval()
        self.policy_picker.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"🤖 Agents: {self.n_agents} (AGVs: {self.agv_ids}, Pickers: {self.picker_ids})")
    
    def _load_model_weights(self):
        """Load model weights"""
        if 'policy_agv_state_dict' in self.checkpoint:
            self.policy_agv.load_state_dict(self.checkpoint['policy_agv_state_dict'])
            self.policy_picker.load_state_dict(self.checkpoint['policy_picker_state_dict'])
        else:
            for key in self.checkpoint.keys():
                if 'agv' in key.lower() and 'state_dict' in key.lower():
                    self.policy_agv.load_state_dict(self.checkpoint[key])
                elif 'picker' in key.lower() and 'state_dict' in key.lower():
                    self.policy_picker.load_state_dict(self.checkpoint[key])
        
        print("✅ Model weights loaded successfully")
    
    def safe_reset(self, env, seed=None):
        """Safe environment reset"""
        reset_result = env.reset(seed=seed) if seed is not None else env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        if isinstance(obs, np.ndarray) and obs.dtype == object:
            obs = list(obs)
        if not isinstance(obs, (list, tuple)):
            obs = [obs]
        
        return obs, info
    
    def safe_step(self, env, actions):
        """Safe environment step"""
        step_result = env.step(tuple(actions))
        
        if len(step_result) == 5:
            next_obs, rewards, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            next_obs, rewards, done, info = step_result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step result format: {len(step_result)} elements")
        
        if isinstance(next_obs, np.ndarray) and next_obs.dtype == object:
            next_obs = list(next_obs)
        if not isinstance(next_obs, (list, tuple)):
            next_obs = [next_obs]
            
        return next_obs, rewards, terminated, truncated, info
    
    def detect_agent_roles(self, env, reset_info, obs_sample):
        """Detect AGV and Picker agents"""
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
    
    def enhance_rewards_like_training(self, original_rewards, info, step, max_steps):
        """Apply the SAME reward enhancements as during training"""
        enhanced_rewards = []
        
        # Extract delivery bonus from config
        delivery_bonus = self.config.get("delivery_bonus", 20.0)
        loading_bonus = self.config.get("loading_bonus", 5.0)
        step_penalty = self.config.get("step_penalty", -0.001)
        efficiency_bonus = self.config.get("efficiency_bonus", 2.0)
        exploration_bonus = self.config.get("exploration_bonus", 0.1)
        
        # Calculate bonuses
        delivery_count = info.get("shelf_deliveries", 0) if info else 0
        total_delivery_bonus = delivery_count * delivery_bonus
        
        efficiency = 0.0
        if step < max_steps * 0.6:
            efficiency = efficiency_bonus
        
        for i, original_reward in enumerate(original_rewards):
            enhanced_reward = original_reward
            
            # Apply the SAME enhancements as in training
            enhanced_reward += total_delivery_bonus / self.n_agents  # Shared delivery bonus
            enhanced_reward += efficiency
            enhanced_reward += exploration_bonus
            enhanced_reward += step_penalty
            
            enhanced_rewards.append(enhanced_reward)
        
        return enhanced_rewards, delivery_count
    
    def evaluate(self):
        print(f"\n🎬 Starting evaluation of {self.num_episodes} episodes...")
        print("💰 Using TRAINING reward conditions (with artificial bonuses)")
        
        episode_rewards = []
        episode_deliveries = []
        episode_lengths = []
        agent_rewards = {f"agent_{i}": [] for i in range(self.n_agents)}
        
        for episode in range(self.num_episodes):
            print(f"\n📊 Episode {episode + 1}/{self.num_episodes}")
            
            obs, info = self.safe_reset(self.env, seed=episode + 1000)
            episode_reward = np.zeros(self.n_agents)
            deliveries = 0
            steps = 0
            
            if self.render:
                print("🖥️  Rendering environment...")
                self.env.render()
                time.sleep(1)
            
            for step in range(self.max_steps):
                actions = []
                
                # Get actions from trained policies with exploration noise
                for agent_id in range(self.n_agents):
                    ob = np.asarray(obs[agent_id], dtype=np.float32)
                    ob_t = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
                    
                    if agent_id in self.agv_ids:
                        with torch.no_grad():
                            logits, value = self.policy_agv(ob_t)
                            noise = torch.randn_like(logits) * 0.1
                            logits = logits + noise
                            probs = torch.softmax(logits, dim=-1)
                            dist = torch.distributions.Categorical(probs)
                            action = dist.sample()
                    else:
                        with torch.no_grad():
                            logits, value = self.policy_picker(ob_t)
                            noise = torch.randn_like(logits) * 0.1
                            logits = logits + noise
                            probs = torch.softmax(logits, dim=-1)
                            dist = torch.distributions.Categorical(probs)
                            action = dist.sample()
                    
                    actions.append(action.item())
                
                # Step environment
                next_obs, rewards, terminated, truncated, info = self.safe_step(self.env, actions)
                
                # Get original rewards
                if isinstance(rewards, (list, tuple)):
                    original_rewards = [float(r) for r in rewards]
                else:
                    original_rewards = [float(rewards)] * self.n_agents
                
                # Apply SAME reward enhancements as in training
                enhanced_rewards, new_deliveries = self.enhance_rewards_like_training(
                    original_rewards, info, step, self.max_steps
                )
                
                if new_deliveries > 0:
                    deliveries += new_deliveries
                    print(f"🎉 Delivery at step {step}! Total: {deliveries}")
                    print(f"   Raw rewards: {original_rewards}")
                    print(f"   Enhanced rewards: {[f'{r:.2f}' for r in enhanced_rewards]}")
                
                # Accumulate ENHANCED rewards (like in training)
                for i in range(self.n_agents):
                    episode_reward[i] += enhanced_rewards[i]
                    agent_rewards[f"agent_{i}"].append(enhanced_rewards[i])
                
                obs = next_obs
                steps += 1
                
                # Render
                if self.render and (step % 20 == 0 or new_deliveries > 0):
                    self.env.render()
                    if new_deliveries > 0:
                        time.sleep(0.5)
                    else:
                        time.sleep(0.05)
                
                # Check termination
                done = False
                if isinstance(terminated, (list, tuple)):
                    done = all(terminated) or (isinstance(truncated, (list, tuple)) and all(truncated))
                else:
                    done = terminated or truncated
                
                if done:
                    if steps < self.max_steps:
                        print(f"🏁 Episode terminated at step {step}")
                    break
            
            total_reward = np.sum(episode_reward)
            episode_rewards.append(total_reward)
            episode_deliveries.append(deliveries)
            episode_lengths.append(steps)
            
            print(f"📈 Episode {episode + 1} Summary:")
            print(f"   Total Enhanced Reward: {total_reward:.2f}")
            print(f"   Deliveries: {deliveries}")
            print(f"   Steps: {steps}")
            print(f"   Agent Enhanced Rewards: {[f'{r:.2f}' for r in episode_reward]}")
        
        # Print final statistics
        self._print_statistics(episode_rewards, episode_deliveries, episode_lengths, agent_rewards)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_deliveries': episode_deliveries,
            'episode_lengths': episode_lengths,
            'agent_rewards': agent_rewards
        }
    
    def _print_statistics(self, episode_rewards, episode_deliveries, episode_lengths, agent_rewards):
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY - TRAINING REWARD CONDITIONS")
        print(f"{'='*60}")
        
        print(f"📈 Enhanced Reward Statistics:")
        print(f"   Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Min Reward: {np.min(episode_rewards):.2f}")
        print(f"   Max Reward: {np.max(episode_rewards):.2f}")
        
        print(f"\n🎯 Delivery Statistics:")
        print(f"   Average Deliveries: {np.mean(episode_deliveries):.2f} ± {np.std(episode_deliveries):.2f}")
        print(f"   Total Deliveries: {sum(episode_deliveries)}")
        success_rate = np.mean([1 if d > 0 else 0 for d in episode_deliveries]) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n⏱️  Episode Length:")
        print(f"   Average Steps: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        print(f"\n🤖 Agent Performance (Enhanced Rewards):")
        for agent_id in range(self.n_agents):
            agent_rew = agent_rewards[f"agent_{agent_id}"]
            agent_type = "AGV" if agent_id in self.agv_ids else "Picker"
            total_agent_reward = np.sum(agent_rew)
            print(f"   {agent_type} {agent_id}: {total_agent_reward:.2f} total reward")
    
    def close(self):
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained TA-RWARE model with training rewards')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--env-id', type=str, default=None,
                       help='Environment ID')
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return
    
    evaluator = TrainingConditionEvaluator(
        model_path=args.model_path,
        env_id=args.env_id,
        render=not args.no_render,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps
    )
    
    try:
        results = evaluator.evaluate()
        print(f"\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()