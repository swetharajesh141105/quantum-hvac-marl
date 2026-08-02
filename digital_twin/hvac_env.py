import gymnasium as gym
from gymnasium import spaces
import numpy as np
from thermal_model import RoomThermalModel

class HVACEnv(gym.Env):
    def __init__(self, config=None):
        self.max_steps = 96
        self.dt = 0.25
        self.comfort_min = 20.0
        self.comfort_max = 25.0
        self.thermal_model = RoomThermalModel({
            "thermal_mass": 5.0,
            "insulation_resistance": 2.0,
            "ac_cooling_power": 3.0,
            "occupancy_heat_gain": 0.5,
        })
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -20.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 1.0, 24.0, 2000.0], dtype=np.float32),
            dtype=np.float32
        )
        self.current_temp = 22.0
        self.current_step = 0

    def _get_current_conditions(self, step):
        hour = (step * self.dt) % 24
        outdoor = 15.0 + 10.0 * np.sin(((hour - 5) * np.pi) / 12)
        occupancy = 1.0 if 9 <= hour <= 18 else 0.0
        return outdoor, occupancy, 400.0 + (300.0 * occupancy)

    def _calculate_reward(self, temp, ac_action):
        penalty = 0.1 * ac_action
        if temp < self.comfort_min:
            penalty += (self.comfort_min - temp) * 2.0
        elif temp > self.comfort_max:
            penalty += (temp - self.comfort_max) * 2.0
        return -penalty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_temp = 22.0
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        outdoor, occupancy, _ = self._get_current_conditions(self.current_step)
        self.current_temp = self.thermal_model.step(
            self.current_temp, outdoor, action, occupancy, self.dt
        )
        self.current_step += 1
        reward = self._calculate_reward(self.current_temp, action)
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        outdoor, occupancy, co2 = self._get_current_conditions(self.current_step)
        return np.array([
            self.current_temp, outdoor, occupancy,
            (self.current_step * self.dt) % 24, co2
        ], dtype=np.float32)
