"""
3 Cooperative Agents: Energy, Comfort, Carbon.
Each is a separate PPO policy trained on the same env but with a specialized reward.
"""
from stable_baselines3 import PPO
from hvac_env import HVACEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from lstm_occupancy import predict_occupancy_30min, OccupancyLSTM


class SingleObjectiveWrapper(gym.Wrapper):
    """Wraps HVACEnv so each agent SPECIALIZES in its objective but still sees
    the full blended reward, preventing reward collapse (e.g. all-OFF forever)."""
    WEIGHTS = {
        "energy":  (0.6, 0.2, 0.2),
        "comfort": (0.2, 0.6, 0.2),
        "carbon":  (0.2, 0.2, 0.6),
    }

    def __init__(self, env, objective):
        super().__init__(env)
        self.objective = objective  # "energy" | "comfort" | "carbon"

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        we, wc, wcarb = self.WEIGHTS[self.objective]
        blended = we * info["energy"] + wc * info["comfort"] + wcarb * info["carbon"]
        return obs, blended, done, truncated, info


ROOM_PARAMS = {
    "thermal_mass": 5.0,
    "insulation_resistance": 2.0,
    "ac_cooling_power": 3.0,
    "occupancy_heat_gain": 0.5,
}

def robust_load(objective, suffix="_agent_federated"):
    """Workaround for a Windows-specific bug where PPO.load() fails to read
    the checkpoint zip via its internal stream-based reader. This manually
    extracts to a real temp file on disk and loads from a plain file path
    instead, which works reliably."""
    import zipfile, tempfile, os
    import torch

    zip_path = f"{objective}{suffix}.zip"
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmpdir)

    env = SingleObjectiveWrapper(HVACEnv(ROOM_PARAMS), objective)
    model = PPO("MlpPolicy", env, verbose=0)
    policy_state = torch.load(os.path.join(tmpdir, "policy.pth"), map_location="cpu", weights_only=True)
    model.policy.load_state_dict(policy_state)
    return model


def train_agent(objective, timesteps=15000, seed=42):
    env = SingleObjectiveWrapper(HVACEnv(ROOM_PARAMS), objective)
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.02, seed=seed)
    model.learn(total_timesteps=timesteps)
    model.save(f"{objective}_agent")
    return model


def joint_decision(temp, outdoor_temp, occupancy, t, agents, co2=400.0,
                    comfort_high=27.0, comfort_low=21.0):
    """Weighted agent voting (Comfort weighted 2x) + hard comfort-safety
    override, so the system stays reliable even if a specific agent's
    training didn't fully converge."""
    obs = np.array([temp, outdoor_temp, occupancy, t, co2], dtype=np.float32)
    votes = {obj: int(agent.predict(obs, deterministic=True)[0]) for obj, agent in agents.items()}
    weighted_score = 2 * votes.get("comfort", 0) + votes.get("energy", 0) + votes.get("carbon", 0)
    action = int(weighted_score >= 2)

    if temp > comfort_high:
        action = 1
    elif temp < comfort_low:
        action = 0
    return action


def joint_decision_verbose(temp, outdoor_temp, occupancy, t, agents, co2=400.0,
                            comfort_high=27.0, comfort_low=21.0):
    """Same logic as joint_decision, but also returns WHY the decision was
    made — per-agent votes, weighted score, and whether the hard comfort
    override fired. Used by xai_explainer.py to generate plain-English
    explanations without duplicating the decision logic."""
    obs = np.array([temp, outdoor_temp, occupancy, t, co2], dtype=np.float32)
    votes = {obj: int(agent.predict(obs, deterministic=True)[0]) for obj, agent in agents.items()}
    weighted_score = 2 * votes.get("comfort", 0) + votes.get("energy", 0) + votes.get("carbon", 0)
    action = int(weighted_score >= 2)
    override = None

    if temp > comfort_high:
        action = 1
        override = "too_hot"
    elif temp < comfort_low:
        action = 0
        override = "too_cold"

    reasoning = {
        "votes": votes,
        "weighted_score": weighted_score,
        "override": override,
        "temp": temp,
        "outdoor_temp": outdoor_temp,
        "occupancy": occupancy,
        "comfort_high": comfort_high,
        "comfort_low": comfort_low,
    }
    return action, reasoning
def joint_decision_with_prediction(temp, outdoor_temp, occupancy_history, t, agents):
    """Same as joint_decision, but uses LSTM-predicted occupancy instead of current occupancy."""
    occupancy_model = OccupancyLSTM()
    occupancy_model.load_state_dict(torch.load("occupancy_lstm.pt"))
    occupancy_model.eval()
    will_occupy, _ = predict_occupancy_30min(occupancy_model, occupancy_history)
    predicted_occupancy = int(will_occupy)
    return joint_decision(temp, outdoor_temp, predicted_occupancy, t, agents)


if __name__ == "__main__":
    agents = {}
    for objective in ["energy", "comfort", "carbon"]:
        print(f"Training {objective} agent...")
        agents[objective] = train_agent(objective)

    print("\nTraining complete. Testing joint decision:")
    action = joint_decision(temp=30.0, outdoor_temp=35.0, occupancy=1, t=0, agents=agents)
    print(f"Joint AC decision: {'ON' if action else 'OFF'}")
