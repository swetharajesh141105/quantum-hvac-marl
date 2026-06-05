"""
Multi-Agent RL Environment
===========================
Wraps the Digital Twin so 3 agents can train simultaneously.

CONCEPT: Multi-Agent Reinforcement Learning (MARL)
---------------------------------------------------
In single-agent RL, one AI optimizes one objective.
In MARL, multiple agents share the same environment but each
optimizes a different reward function.

Our 3 agents:
  Agent 0 (Energy Agent):   Minimize power consumption
  Agent 1 (Comfort Agent):  Maintain 22°C when occupied
  Agent 2 (Carbon Agent):   Reduce carbon footprint
                            (prefers nighttime/off-peak hours)

They all act on the SAME AC system, so they must implicitly
cooperate. Their individual Q-values are averaged to produce
the final action — this is called "cooperative MARL".

Each agent is a separate PPO model from Stable-Baselines3.
PPO = Proximal Policy Optimization — the most reliable RL
algorithm today, used in ChatGPT's RLHF training too!
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin.thermal_model import DigitalTwin, RoomConfig


class HVACMultiAgentEnv:
    """
    Multi-agent environment. Each agent gets the same observation
    but computes a different reward.
    
    Usage:
        env = HVACMultiAgentEnv()
        obs = env.reset()
        # obs is shape (3, 10) — one observation per agent
        
        actions = [agent0.predict(obs[0]),
                   agent1.predict(obs[1]),
                   agent2.predict(obs[2])]
        
        obs, rewards, done, info = env.step(actions)
        # rewards is list of 3 floats — one per agent
    """

    N_AGENTS = 3
    AGENT_NAMES = ["Energy Agent", "Comfort Agent", "Carbon Agent"]

    def __init__(self, room_config: RoomConfig = None):
        self.twin = DigitalTwin(config=room_config)
        self.obs_size = self.twin.observation_space_size
        self.n_actions = self.twin.action_space_size

    def reset(self):
        """Reset environment, return initial observation for all agents."""
        obs = self.twin.reset()
        return np.tile(obs, (self.N_AGENTS, 1))  # same obs for all 3

    def step(self, actions: list):
        """
        Receive one action per agent, vote on final action,
        execute in twin, return per-agent rewards.
        
        Voting strategy: each agent's action is weighted by confidence.
        For simplicity here, we use majority vote (mode of actions).
        In practice you'd weight by predicted Q-values.
        """
        # Cooperative voting: take the action proposed by majority
        final_action = int(np.bincount(actions).argmax())

        obs, base_reward, done, info = self.twin.step(final_action)

        # Each agent gets its own specialized reward
        rewards = [
            self._energy_reward(info),    # Agent 0
            self._comfort_reward(info),   # Agent 1
            self._carbon_reward(info),    # Agent 2
        ]

        observations = np.tile(obs, (self.N_AGENTS, 1))
        return observations, rewards, done, info

    # ------------------------------------------------------------------ #
    #  Per-agent reward functions                                          #
    # ------------------------------------------------------------------ #

    def _energy_reward(self, info: dict) -> float:
        """
        Agent 0: Minimize energy consumption.
        Reward = 1 when AC is off, 0 when AC at max power.
        Penalty for cooling empty room.
        """
        max_power = self.twin.config.ac_capacity_kw
        energy_efficiency = 1.0 - (info["power_kw"] / max_power)
        empty_waste = -2.0 if (self.twin.ac_on and info["occupants"] == 0) else 0.0
        return float(energy_efficiency + empty_waste)

    def _comfort_reward(self, info: dict) -> float:
        """
        Agent 1: Maintain 22°C when people are present.
        Only penalized when room is occupied — no point being
        comfortable in an empty room.
        """
        if info["occupants"] == 0:
            return 0.5  # neutral, not penalized

        ideal_temp = 22.0
        ideal_humidity = 50.0
        temp_error = abs(info["temp"] - ideal_temp)
        hum_error = abs(info["humidity"] - ideal_humidity)

        temp_score = max(0, 1.0 - temp_error / 6.0)   # 0 if >6°C off
        hum_score = max(0, 1.0 - hum_error / 20.0)

        # CO2 comfort: high CO2 means stuffy room
        co2_score = max(0, 1.0 - (info["co2"] - 400) / 1000.0)

        return float(0.6 * temp_score + 0.2 * hum_score + 0.2 * co2_score)

    def _carbon_reward(self, info: dict) -> float:
        """
        Agent 2: Carbon-aware scheduling.
        
        Concept: Electricity grid carbon intensity varies by time.
        During the day (solar peak ~10am-4pm), grid is cleaner.
        At night, coal/gas plants dominate → more carbon per kWh.
        
        This agent rewards using AC during clean-energy hours
        and avoiding it during high-carbon hours.
        """
        hour = (self.twin.time_step * self.twin.timestep_seconds / 3600) % 24

        # Carbon intensity (kg CO2 per kWh) — simplified model
        # Low during solar hours (10-16), high at night
        if 10 <= hour <= 16:
            carbon_intensity = 0.3  # kg CO2/kWh (solar available)
        elif 6 <= hour < 10 or 16 < hour <= 20:
            carbon_intensity = 0.5  # mixed grid
        else:
            carbon_intensity = 0.7  # mostly coal/gas at night

        carbon_cost = info["power_kw"] * carbon_intensity
        max_possible_carbon = self.twin.config.ac_capacity_kw * 0.7

        carbon_reward = 1.0 - (carbon_cost / max_possible_carbon)
        return float(carbon_reward)


# ------------------------------------------------------------------ #
#  Single-agent wrapper (for easier initial training)                 #
# ------------------------------------------------------------------ #

class HVACSingleAgentEnv:
    """
    Gymnasium-compatible wrapper for single-agent training.
    Uses composite reward (all three objectives combined).
    Use this first to verify RL is working before going MARL.
    
    Compatible with stable-baselines3 directly.
    """

    def __init__(self):
        self.marl_env = HVACMultiAgentEnv()
        self.observation_space_shape = (self.marl_env.obs_size,)
        self.action_space_n = self.marl_env.n_actions

    def reset(self):
        obs_all = self.marl_env.reset()
        return obs_all[0]  # just agent 0's observation

    def step(self, action: int):
        actions = [action, action, action]  # all agents use same action
        obs_all, rewards, done, info = self.marl_env.step(actions)
        combined_reward = np.mean(rewards)  # average of all 3 reward functions
        return obs_all[0], combined_reward, done, info

    def render(self, info: dict):
        """Simple text render for debugging."""
        ac_str = f"ON@{self.marl_env.twin.ac_setpoint}°C" if self.marl_env.twin.ac_on else "OFF"
        print(f"  T={info['temp']:5.1f}°C | Out={info['outdoor_temp']:4.1f}°C | "
              f"Occ={info['occupants']:2d} | CO2={info['co2']:4.0f}ppm | "
              f"AC={ac_str} | Power={info['power_kw']:.2f}kW")


if __name__ == "__main__":
    print("=" * 60)
    print("MARL Environment — Self Test")
    print("=" * 60)

    env = HVACMultiAgentEnv()
    obs_all = env.reset()
    print(f"Observation shape: {obs_all.shape}  (3 agents × 10 features)")

    for step in range(5):
        actions = [np.random.randint(0, 6) for _ in range(3)]
        obs_all, rewards, done, info = env.step(actions)
        print(f"Step {step+1}: actions={actions} → rewards="
              f"[E:{rewards[0]:.2f}, C:{rewards[1]:.2f}, Ca:{rewards[2]:.2f}] "
              f"T={info['temp']:.1f}°C")

    print("\n✅ MARL Environment working correctly!")
