"""
3 Cooperative Agents: Energy, Comfort, Carbon.
Each is a separate PPO policy trained on the same env but with a specialized reward.
"""
from stable_baselines3 import PPO
from hvac_env import HVACEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SingleObjectiveWrapper(gym.Wrapper):
    """Wraps HVACEnv so one agent only sees its own reward component."""
    def __init__(self, env, objective):
        super().__init__(env)
        self.objective = objective  # "energy" | "comfort" | "carbon"

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, info[self.objective], done, truncated, info

ROOM_PARAMS = {
    "thermal_mass": 5.0,
    "insulation_resistance": 2.0,
    "ac_cooling_power": 3.0,
    "occupancy_heat_gain": 0.5,
}

def train_agent(objective, timesteps=5000):
    env = SingleObjectiveWrapper(HVACEnv(ROOM_PARAMS), objective)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(f"{objective}_agent")
    return model

def joint_decision(temp, outdoor_temp, occupancy, t, agents):
    """Simple voting: majority of agents decide ON/OFF."""
    obs = np.array([temp, outdoor_temp, occupancy, t], dtype=np.float32)
    votes = [agent.predict(obs, deterministic=True)[0] for agent in agents.values()]
    return int(sum(votes) >= 2)  # majority vote

if __name__ == "__main__":
    agents = {}
    for objective in ["energy", "comfort", "carbon"]:
        print(f"Training {objective} agent...")
        agents[objective] = train_agent(objective)

    print("\nTraining complete. Testing joint decision:")
    action = joint_decision(temp=30.0, outdoor_temp=35.0, occupancy=1, t=0, agents=agents)
    print(f"Joint AC decision: {'ON' if action else 'OFF'}")