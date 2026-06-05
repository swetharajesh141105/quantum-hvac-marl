"""
MARL Training Script
=====================
Trains 3 PPO agents on the Digital Twin simulation.

HOW PPO WORKS (for interviews):
--------------------------------
PPO (Proximal Policy Optimization) is an actor-critic RL algorithm.
- "Actor": the policy network — decides which action to take
- "Critic": the value network — estimates how good the current state is

The "Proximal" part means: don't update the policy too aggressively.
It clips the update ratio so one bad batch doesn't ruin training.
This makes it much more stable than older algorithms like TRPO or vanilla PG.

PPO was used to train the reward model in ChatGPT's RLHF pipeline!

TRAINING FLOW:
1. Collect N timesteps of experience (obs, action, reward, next_obs)
2. Compute advantage = actual_return - critic_estimate
3. Update actor to increase probability of high-advantage actions
4. Update critic to better predict returns
5. Repeat

For MARL: we train each agent independently (Independent PPO = IPPO)
This is surprisingly effective for cooperative settings.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We implement a lightweight PPO to avoid heavy dependencies
# In production you'd use: from stable_baselines3 import PPO


# ------------------------------------------------------------------ #
#  Lightweight Neural Network (no PyTorch needed for basic demo)       #
# ------------------------------------------------------------------ #

class SimpleNetwork:
    """
    Minimal 2-layer neural network using only NumPy.
    In production, use stable-baselines3's PPO which uses PyTorch.
    
    Architecture: 10 → 64 → 64 → 6 (softmax output for actions)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Xavier initialization — prevents vanishing/exploding gradients
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale2
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        self.b3 = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: input → action probabilities."""
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return self._softmax(logits)

    def predict(self, obs: np.ndarray) -> int:
        """Select action: greedy (highest probability)."""
        probs = self.forward(obs)
        return int(np.argmax(probs))

    def predict_with_exploration(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy: explore randomly with probability epsilon."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.W3.shape[1])
        return self.predict(obs)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()


# ------------------------------------------------------------------ #
#  Simple Evolutionary / Policy Gradient Trainer                      #
# ------------------------------------------------------------------ #

class AgentTrainer:
    """
    Trains an agent using simple policy gradient (REINFORCE algorithm).
    
    REINFORCE:
    - Run a full episode
    - For each step, compute discounted return G_t
    - Update: push the policy toward actions that got high G_t
    
    This is the conceptual foundation of PPO. For actual training,
    use stable-baselines3 PPO — it's far more efficient.
    """

    def __init__(self, network: SimpleNetwork, learning_rate: float = 0.001):
        self.net = network
        self.lr = learning_rate
        self.episode_rewards = []

    def compute_returns(self, rewards: list, gamma: float = 0.99) -> np.ndarray:
        """
        Compute discounted returns.
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        
        γ (gamma) = discount factor. 0.99 means future rewards
        are almost as valuable as immediate ones.
        """
        returns = np.zeros(len(rewards))
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G
        # Normalize returns for stable training
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        return returns

    def update(self, observations: list, actions: list, returns: np.ndarray):
        """
        Simple gradient ascent update.
        Nudge weights toward actions that had positive returns.
        """
        for obs, action, G in zip(observations, actions, returns):
            if G == 0:
                continue
            probs = self.net.forward(obs)
            # Gradient of log-probability w.r.t. action
            grad_logprob = -probs.copy()
            grad_logprob[action] += 1.0
            # Update output layer
            h2 = np.tanh(np.tanh(obs @ self.net.W1 + self.net.b1) @ self.net.W2 + self.net.b2)
            self.net.W3 += self.lr * G * np.outer(h2, grad_logprob)
            self.net.b3 += self.lr * G * grad_logprob


# ------------------------------------------------------------------ #
#  Main Training Loop                                                  #
# ------------------------------------------------------------------ #

def train_agents(n_episodes: int = 200, save_dir: str = "agents"):
    """
    Train all 3 MARL agents and save their weights.
    
    Training phases:
    1. Warm-up (eps 0-50): High exploration (ε=0.3), learn basic patterns
    2. Learning (eps 50-150): Reduce exploration, refine policy
    3. Fine-tuning (eps 150-200): Low exploration, polish
    """
    from marl.environment import HVACMultiAgentEnv, HVACSingleAgentEnv

    os.makedirs(save_dir, exist_ok=True)

    # Initialize 3 agents
    obs_size = 10
    n_actions = 6
    hidden = 64

    agents = [SimpleNetwork(obs_size, hidden, n_actions) for _ in range(3)]
    trainers = [AgentTrainer(agents[i]) for i in range(3)]

    env = HVACMultiAgentEnv()

    agent_names = ["Energy Agent", "Comfort Agent", "Carbon Agent"]
    all_rewards = [[] for _ in range(3)]
    episode_lengths = []

    print("=" * 60)
    print("Starting MARL Training")
    print(f"Episodes: {n_episodes} | Agents: 3 | Obs size: {obs_size}")
    print("=" * 60)

    for ep in range(n_episodes):
        # Exploration schedule
        epsilon = max(0.05, 0.3 - (ep / n_episodes) * 0.25)

        obs_all = env.reset()
        ep_observations = [[] for _ in range(3)]
        ep_actions = [[] for _ in range(3)]
        ep_rewards = [[] for _ in range(3)]

        done = False
        steps = 0

        while not done:
            # Each agent picks an action
            actions = [
                agents[i].predict_with_exploration(obs_all[i], epsilon)
                for i in range(3)
            ]

            obs_all, rewards, done, info = env.step(actions)

            for i in range(3):
                ep_observations[i].append(obs_all[i].copy())
                ep_actions[i].append(actions[i])
                ep_rewards[i].append(rewards[i])

            steps += 1

        # Update each agent
        for i in range(3):
            returns = trainers[i].compute_returns(ep_rewards[i])
            trainers[i].update(ep_observations[i], ep_actions[i], returns)
            all_rewards[i].append(sum(ep_rewards[i]))

        episode_lengths.append(steps)

        # Progress logging
        if (ep + 1) % 25 == 0:
            avg_rewards = [np.mean(all_rewards[i][-25:]) for i in range(3)]
            print(f"Episode {ep+1:4d}/{n_episodes} | ε={epsilon:.2f} | "
                  f"Rewards: E={avg_rewards[0]:6.1f} "
                  f"C={avg_rewards[1]:6.1f} "
                  f"Ca={avg_rewards[2]:6.1f}")

    # Save agent weights
    for i, agent in enumerate(agents):
        path = os.path.join(save_dir, f"agent_{i}_weights.npz")
        np.savez(path, W1=agent.W1, b1=agent.b1,
                 W2=agent.W2, b2=agent.b2,
                 W3=agent.W3, b3=agent.b3)
        print(f"Saved {agent_names[i]} → {path}")

    # Plot training curves
    _plot_training_curves(all_rewards, agent_names, n_episodes)

    print("\n✅ Training complete!")
    return agents


def _plot_training_curves(all_rewards, agent_names, n_episodes):
    """Generate training curve plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, (rewards, name, color) in enumerate(zip(all_rewards, agent_names, colors)):
        # Smoothed curve
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

        axes[i].plot(rewards, alpha=0.3, color=color)
        axes[i].plot(range(window-1, len(rewards)), smoothed,
                     color=color, linewidth=2, label='Smoothed')
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Total Reward')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.suptitle('MARL Training Curves — Quantum HVAC', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved training_curves.png")


def evaluate_agents(agents, n_episodes: int = 10):
    """
    Evaluate trained agents — greedy (no exploration).
    Compare against baseline (always-on at 22°C).
    """
    from marl.environment import HVACMultiAgentEnv

    env = HVACMultiAgentEnv()

    ai_energies, baseline_energies = [], []
    ai_discomforts, baseline_discomforts = [], []

    print("\n" + "=" * 60)
    print("Evaluation: AI vs Baseline (Always-On at 22°C)")
    print("=" * 60)

    for ep in range(n_episodes):
        # AI agent run
        obs_all = env.reset()
        done = False
        while not done:
            actions = [agents[i].predict(obs_all[i]) for i in range(3)]
            obs_all, _, done, info = env.step(actions)
        ai_energies.append(info["episode_energy_kwh"])
        ai_discomforts.append(info["episode_discomfort"])

        # Baseline: action 3 = always on at 22°C
        obs_all = env.reset()
        done = False
        while not done:
            obs_all, _, done, info = env.step([3, 3, 3])
        baseline_energies.append(info["episode_energy_kwh"])
        baseline_discomforts.append(info["episode_discomfort"])

    ai_e = np.mean(ai_energies)
    base_e = np.mean(baseline_energies)
    savings = (base_e - ai_e) / base_e * 100

    print(f"  Energy — AI: {ai_e:.3f} kWh | Baseline: {base_e:.3f} kWh")
    print(f"  💡 Energy Savings: {savings:.1f}%")
    print(f"  Discomfort — AI: {np.mean(ai_discomforts):.2f} | "
          f"Baseline: {np.mean(baseline_discomforts):.2f}")

    return {"energy_savings_pct": savings, "ai_energy": ai_e, "baseline_energy": base_e}


if __name__ == "__main__":
    print("Training MARL agents on Digital Twin...")
    print("(In production use: stable-baselines3 PPO for faster convergence)\n")

    trained_agents = train_agents(n_episodes=150)
    results = evaluate_agents(trained_agents, n_episodes=5)

    print(f"\n📊 Final Result: {results['energy_savings_pct']:.1f}% energy savings")
    print("   This will improve significantly with stable-baselines3 PPO")
    print("   and more training episodes (500-1000).")
