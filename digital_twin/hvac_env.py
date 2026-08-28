import numpy as np
import gymnasium as gym
from gymnasium import spaces
from thermal_model import RoomThermalModel

class HVACEnv(gym.Env):
    def __init__(self, room_params, episode_length=24, dt=1.0):
        super().__init__()
        self.model = RoomThermalModel(room_params)
        self.episode_length = episode_length
        self.dt = dt
        self.observation_space = spaces.Box(low=0, high=50, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.temp = 30.0
        self.outdoor_temp = 35.0
        self.occupancy = 1
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.temp, self.outdoor_temp, self.occupancy, self.t], dtype=np.float32)

    def step(self, action):
        self.temp = self.model.step(self.temp, self.outdoor_temp, action, self.occupancy, self.dt)
        self.t += 1
        done = self.t >= self.episode_length

        # normalized, balanced reward components (fixes reward collapse)
        energy_penalty = -1.0 * action
        comfort_penalty = -np.clip(abs(self.temp - 24.0) / 5.0, 0, 2.0)
        carbon_penalty = -0.3 * action

        reward = (0.4 * energy_penalty) + (0.4 * comfort_penalty) + (0.2 * carbon_penalty)

        return self._get_obs(), reward, done, False, {
            "energy": energy_penalty, "comfort": comfort_penalty, "carbon": carbon_penalty
        }