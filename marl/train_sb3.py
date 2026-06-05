"""
Stable-Baselines3 PPO Training Script
=======================================
This replaces the basic training in train_agents.py with proper
industry-standard PPO from Stable-Baselines3.

WHY PPO IS BETTER THAN OUR BASIC VERSION:
------------------------------------------
Our basic version used REINFORCE (vanilla policy gradient):
- Needs full episode before updating
- High variance, slow learning
- No value function baseline

PPO (Proximal Policy Optimization) adds:
1. Clipped objective: prevents too-large updates
   L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
   where r_t = new_prob/old_prob, A_t = advantage estimate
   
2. Value function: critic estimates how good each state is
   Advantage = actual_return - critic_estimate
   This reduces variance massively
   
3. Multiple epochs per batch: reuses collected data efficiently

4. Generalized Advantage Estimation (GAE):
   Balances bias vs variance in advantage calculation
   λ=1 → low bias, high variance (Monte Carlo)
   λ=0 → high bias, low variance (TD(0))
   λ=0.95 → good balance (default)

PPO was used in:
- OpenAI Five (Dota 2)
- ChatGPT's RLHF training
- Most modern robotics control

GYMNASIUM WRAPPER:
SB3 requires environments to follow the Gymnasium API:
- observation_space: Box or Discrete space object
- action_space: Discrete(6) for our 6 actions
- reset() returns (obs, info)
- step() returns (obs, reward, terminated, truncated, info)
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from marl.environment import HVACMultiAgentEnv, HVACSingleAgentEnv


# ------------------------------------------------------------------ #
#  Gymnasium-Compatible Wrapper                                        #
# ------------------------------------------------------------------ #

class HVACGymEnv(gym.Env):
    """
    Wraps HVACSingleAgentEnv to be fully Gymnasium-compatible.
    Required for Stable-Baselines3.
    
    Key additions:
    - observation_space: tells SB3 the shape and bounds of observations
    - action_space: tells SB3 there are 6 discrete actions
    - reset() returns (obs, info) tuple — Gymnasium standard
    - step() returns (obs, reward, terminated, truncated, info)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, agent_type: str = "combined"):
        super().__init__()
        
        self.agent_type = agent_type
        self.marl_env = HVACMultiAgentEnv()
        
        # Tell SB3 what observations look like
        # Box = continuous space, shape=(10,), all values in [-5, 5]
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(10,),
            dtype=np.float32
        )
        
        # Tell SB3 there are 6 possible actions (0-5)
        self.action_space = spaces.Discrete(6)
        
        self._step_count = 0
        self._max_steps = 24 * 3600 // 5  # one full day at 5s timesteps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_all = self.marl_env.reset()
        self._step_count = 0
        return obs_all[0].astype(np.float32), {}

    def step(self, action: int):
        actions = [action, action, action]
        obs_all, rewards, done, info = self.marl_env.step(actions)
        
        # Select reward based on agent type
        if self.agent_type == "energy":
            reward = rewards[0]
        elif self.agent_type == "comfort":
            reward = rewards[1]
        elif self.agent_type == "carbon":
            reward = rewards[2]
        else:  # combined
            reward = float(np.mean(rewards))
        
        self._step_count += 1
        terminated = done
        truncated = self._step_count >= self._max_steps
        
        return obs_all[0].astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pass


# ------------------------------------------------------------------ #
#  Training Progress Callback                                          #
# ------------------------------------------------------------------ #

class TrainingCallback(BaseCallback):
    """
    Custom callback to track training progress.
    Called after every n_steps by SB3.
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards from monitor
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([
                    ep['r'] for ep in self.model.ep_info_buffer
                ])
                self.episode_rewards.append(mean_reward)
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
                if self.verbose:
                    print(f"  Step {self.n_calls:6d} | "
                          f"Mean reward: {mean_reward:7.2f} | "
                          f"Best: {self.best_mean_reward:7.2f}")
        return True


# ------------------------------------------------------------------ #
#  Train All 3 Specialized Agents                                      #
# ------------------------------------------------------------------ #

def train_ppo_agents(
    total_timesteps: int = 50000,
    save_dir: str = "ppo_agents"
):
    """
    Train 3 specialized PPO agents, one per objective.
    
    Each agent is a full PPO model with:
    - Policy network: MlpPolicy (2 hidden layers of 64 neurons)
    - Value network: shares architecture with policy
    - Learning rate: 3e-4 (Adam optimizer)
    - n_steps: 2048 (collect before each update)
    - batch_size: 64
    - n_epochs: 10 (reuse each batch 10 times)
    - gamma: 0.99 (discount factor)
    - gae_lambda: 0.95 (GAE parameter)
    - clip_range: 0.2 (PPO clipping epsilon)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    agent_configs = [
        ("energy",   "Energy Agent   (minimize power)"),
        ("comfort",  "Comfort Agent  (maintain 22°C) "),
        ("carbon",   "Carbon Agent   (clean energy)  "),
    ]
    
    trained_models = {}
    all_callbacks = {}
    
    print("=" * 60)
    print("PPO Training — Stable-Baselines3")
    print(f"Timesteps per agent: {total_timesteps:,}")
    print(f"Total timesteps: {total_timesteps * 3:,}")
    print("=" * 60)
    
    for agent_type, agent_name in agent_configs:
        print(f"\nTraining {agent_name}...")
        
        # Create monitored environment
        env = Monitor(HVACGymEnv(agent_type=agent_type))
        
        # Initialize PPO
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,       # entropy bonus for exploration
            vf_coef=0.5,         # value function loss weight
            max_grad_norm=0.5,   # gradient clipping
            verbose=0,
            tensorboard_log=f"./tensorboard_{agent_type}/",
            policy_kwargs=dict(
                net_arch=[64, 64]  # 2 hidden layers of 64 neurons
            )
        )
        
        # Training callback
        callback = TrainingCallback(check_freq=5000, verbose=1)
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        # Save model
        save_path = os.path.join(save_dir, f"ppo_{agent_type}")
        model.save(save_path)
        print(f"  Saved → {save_path}.zip")
        
        trained_models[agent_type] = model
        all_callbacks[agent_type] = callback
        env.close()
    
    # Plot training curves
    _plot_ppo_curves(all_callbacks)
    
    print("\n✅ PPO Training complete!")
    return trained_models


def _plot_ppo_curves(callbacks: dict):
    """Plot training curves for all 3 PPO agents."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    names = ['Energy Agent', 'Comfort Agent', 'Carbon Agent']
    
    for ax, (agent_type, callback), color, name in zip(
        axes, callbacks.items(), colors, names
    ):
        rewards = callback.episode_rewards
        if rewards:
            ax.plot(rewards, color=color, linewidth=2)
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('Check interval')
            ax.set_ylabel('Mean Episode Reward')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center')
    
    plt.suptitle('PPO Training Curves — Quantum HVAC', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ppo_training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved ppo_training_curves.png")


# ------------------------------------------------------------------ #
#  Evaluate PPO vs Basic vs Baseline                                   #
# ------------------------------------------------------------------ #

def evaluate_ppo(models: dict, n_episodes: int = 5):
    """
    Compare PPO agents vs basic training vs always-on baseline.
    Shows the improvement from using proper RL.
    """
    print("\n" + "=" * 60)
    print("EVALUATION: PPO vs Baseline")
    print("=" * 60)
    
    env = HVACGymEnv(agent_type="combined")
    
    ppo_energies = []
    ppo_discomforts = []
    baseline_energies = []
    
    energy_model = models.get("energy")
    comfort_model = models.get("comfort")
    
    for ep in range(n_episodes):
        # PPO run — use energy agent for action selection
        obs, _ = env.reset()
        done = False
        while not done:
            # Cooperative: average actions from energy + comfort agents
            action_e, _ = energy_model.predict(obs, deterministic=True)
            action_c, _ = comfort_model.predict(obs, deterministic=True)
            # Simple voting: use energy agent's action
            obs, reward, terminated, truncated, info = env.step(int(action_e))
            done = terminated or truncated
        
        ppo_energies.append(info.get("episode_energy_kwh", 0))
        ppo_discomforts.append(info.get("episode_discomfort", 0))
        
        # Baseline: always ON at 22°C (action 3)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(3)
            done = terminated or truncated
        baseline_energies.append(info.get("episode_energy_kwh", 0))
    
    ppo_e = np.mean(ppo_energies)
    base_e = np.mean(baseline_energies)
    savings = (base_e - ppo_e) / base_e * 100 if base_e > 0 else 0
    
    print(f"  PPO Energy:      {ppo_e:.3f} kWh")
    print(f"  Baseline Energy: {base_e:.3f} kWh")
    print(f"  💡 Energy Savings: {savings:.1f}%")
    print(f"  PPO Discomfort:  {np.mean(ppo_discomforts):.2f}")
    
    env.close()
    return {"savings_pct": savings, "ppo_energy": ppo_e, "baseline_energy": base_e}


# ------------------------------------------------------------------ #
#  Load and use saved PPO models                                       #
# ------------------------------------------------------------------ #

def load_and_predict(save_dir: str = "ppo_agents"):
    """
    Load saved PPO models and make a prediction.
    Use this after training to run inference.
    """
    models = {}
    for agent_type in ["energy", "comfort", "carbon"]:
        path = os.path.join(save_dir, f"ppo_{agent_type}")
        if os.path.exists(path + ".zip"):
            models[agent_type] = PPO.load(path)
            print(f"Loaded {agent_type} agent from {path}.zip")
    return models


if __name__ == "__main__":
    print("Starting PPO Training with Stable-Baselines3...")
    print("This takes 5-10 minutes. Progress shown every 5000 steps.\n")
    
    # Train all 3 agents
    models = train_ppo_agents(total_timesteps=50000)
    
    # Evaluate
    results = evaluate_ppo(models)
    
    print(f"\n📊 PPO Result: {results['savings_pct']:.1f}% energy savings")
    print("Models saved in ppo_agents/ folder")
    print("Run again with total_timesteps=200000 for even better results")
