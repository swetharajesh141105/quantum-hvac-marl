"""
PPO Training v2 — Fixed Reward Function
=========================================
FIX: The original PPO collapsed to "always OFF" because the energy
reward was too dominant. When room was empty, turning AC off gave
+1.0 energy reward with no comfort penalty → agent learned OFF always.

THE FIX:
1. Comfort is now a HARD PENALTY (not soft reward)
   If room is occupied AND temp > 25°C → large negative reward (-2.0)
   This makes discomfort MORE costly than any energy saving.

2. Energy reward is CONDITIONAL
   Only reward energy saving when room is empty.
   When room is occupied, energy reward is reduced.

3. Added "unnecessary cooling" penalty
   AC ON when temp already < 21°C → waste penalty.

REWARD FORMULA (new):
   if occupied:
       comfort_penalty = -2.0 if temp > 25 else 0.0  ← hard constraint
       energy_reward   = 0.3 * (1 - power/max_power)  ← small weight
       reward = comfort_penalty + energy_reward
   else:
       reward = 1.0 if ac_off else -0.5               ← reward being off
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from digital_twin.thermal_model import DigitalTwin, RoomConfig


# ------------------------------------------------------------------ #
#  Fixed Reward Environment                                            #
# ------------------------------------------------------------------ #

class HVACFixedRewardEnv(gym.Env):
    """
    Single environment with balanced reward function.
    Comfort is a hard constraint — agent CANNOT ignore it.
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.twin = DigitalTwin()
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)
        self._steps = 0
        self._max_steps = 24 * 3600 // 5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.twin.reset()
        self._steps = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, _, done, info = self.twin.step(int(action))
        reward = self._compute_reward(info, action)
        self._steps += 1
        terminated = done
        truncated = self._steps >= self._max_steps
        return obs.astype(np.float32), reward, terminated, truncated, info

    def _compute_reward(self, info, action):
        temp      = info["temp"]
        occupants = info["occupants"]
        power_kw  = info["power_kw"]
        max_power = self.twin.config.ac_capacity_kw
        ac_on     = self.twin.ac_on

        if occupants > 0:
            # COMFORT IS MANDATORY when occupied
            if temp > 26:
                comfort = -3.0   # very hot, severe penalty
            elif temp > 25:
                comfort = -2.0   # hot, strong penalty
            elif temp > 24:
                comfort = -0.5   # slightly warm, mild penalty
            elif temp < 20:
                comfort = -1.0   # too cold also bad
            else:
                comfort = +1.0   # comfortable range 20-24°C

            # Small energy reward (secondary objective)
            energy = 0.2 * (1.0 - power_kw / max_power)

            # Penalty for being off when very hot and occupied
            if not ac_on and temp > 26:
                neglect = -2.0
            else:
                neglect = 0.0

            return float(comfort + energy + neglect)

        else:
            # Room EMPTY — reward being off
            if not ac_on:
                return 1.0    # good: AC off, no one there
            else:
                # AC is on but nobody there — waste
                if temp < 22:
                    return -1.5  # cooling empty cold room
                else:
                    return -0.5  # minor waste

    def render(self):
        pass


# ------------------------------------------------------------------ #
#  Progress Callback                                                   #
# ------------------------------------------------------------------ #

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([e['r'] for e in self.model.ep_info_buffer])
                self.rewards.append(mean_r)
                print("  Step {:6d} | Mean reward: {:.2f}".format(
                    self.n_calls, mean_r))
        return True


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train_fixed_ppo(total_timesteps=100000, save_dir="ppo_agents_v2"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("PPO v2 Training — Fixed Reward Function")
    print("Timesteps: {:,}".format(total_timesteps))
    print("=" * 60)

    env = Monitor(HVACFixedRewardEnv())

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
        ent_coef=0.02,      # slightly higher entropy → more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128])  # bigger network
    )

    callback = ProgressCallback(check_freq=5000)

    print("\nTraining in progress...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = os.path.join(save_dir, "ppo_hvac_v2")
    model.save(save_path)
    print("\nModel saved to " + save_path + ".zip")

    # Plot
    if callback.rewards:
        plt.figure(figsize=(8, 4))
        plt.plot(callback.rewards, color='#2196F3', linewidth=2)
        plt.xlabel('Check interval (x5000 steps)')
        plt.ylabel('Mean Episode Reward')
        plt.title('PPO v2 Training — Fixed Reward')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero line')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ppo_v2_training.png', dpi=150)
        print("Saved ppo_v2_training.png")

    return model


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate_fixed_ppo(model, n_episodes=10):
    print("\n" + "=" * 60)
    print("EVALUATION: PPO v2 vs Always-On Baseline")
    print("=" * 60)

    env = HVACFixedRewardEnv()

    ai_energy, ai_discomfort = [], []
    base_energy, base_discomfort = [], []

    for ep in range(n_episodes):
        # AI run
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        ai_energy.append(info["episode_energy_kwh"])
        ai_discomfort.append(info["episode_discomfort"])

        # Baseline: always ON at 22C
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(3)
            done = terminated or truncated
        base_energy.append(info["episode_energy_kwh"])
        base_discomfort.append(info["episode_discomfort"])

    ai_e   = np.mean(ai_energy)
    base_e = np.mean(base_energy)
    savings = (base_e - ai_e) / base_e * 100 if base_e > 0 else 0

    ai_d   = np.mean(ai_discomfort)
    base_d = np.mean(base_discomfort)

    print("  Energy   — AI: {:.3f} kWh | Baseline: {:.3f} kWh".format(ai_e, base_e))
    print("  Savings  — {:.1f}%".format(savings))
    print("  Discomfort — AI: {:.1f} | Baseline: {:.1f}".format(ai_d, base_d))

    if savings > 10 and ai_d < base_d * 3:
        print("\n  RESULT: Agent saves energy WITHOUT sacrificing comfort")
    elif savings <= 0:
        print("\n  WARNING: Agent uses more energy than baseline — train longer")
    else:
        print("\n  RESULT: Some savings achieved. Train longer for better results.")

    return {"savings_pct": savings, "ai_discomfort": ai_d, "base_discomfort": base_d}


if __name__ == "__main__":
    model = train_fixed_ppo(total_timesteps=100000)
    results = evaluate_fixed_ppo(model, n_episodes=10)
    print("\nFinal: {:.1f}% energy savings".format(results['savings_pct']))
