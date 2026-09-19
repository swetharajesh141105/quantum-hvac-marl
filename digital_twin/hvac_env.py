import numpy as np
import gymnasium as gym
from gymnasium import spaces
from thermal_model import RoomThermalModel
from data_gen import generate_day_profile, load_config


class HVACEnv(gym.Env):
    """
    Uses data_gen.generate_day_profile() for occupancy + outdoor temp so
    TRAINING sees the same realistic office-hours pattern (9am-6pm occupied,
    day/night temp cycle) as the final simulation in run_full_pipeline.py.
    """
    def __init__(self, room_params=None, max_steps=24, dt=1.0, seed=None):
        super().__init__()
        if room_params is None:
            room_params = {"thermal_mass": 5.0, "insulation_resistance": 2.0,
                            "ac_cooling_power": 3.0, "occupancy_heat_gain": 0.5}
        self.model = RoomThermalModel(room_params)
        self.max_steps = max_steps
        self.dt = dt
        self.cfg = load_config()
        self._seed = seed
        self.observation_space = spaces.Box(low=0, high=2000, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        use_seed = seed if seed is not None else self._seed
        profile = generate_day_profile(seed=use_seed)

        outdoor_arr = np.array(profile["outdoor_temp"])
        occ_arr = np.array(profile["occupancy"])
        n = len(outdoor_arr)
        idx = np.linspace(0, n - 1, self.max_steps).astype(int)
        self._outdoor_temp_series = outdoor_arr[idx]
        self._occupancy_series = occ_arr[idx]

        self.t = 0
        self.temp = 26.0
        self.outdoor_temp = float(self._outdoor_temp_series[0])
        self.occupancy = int(self._occupancy_series[0])
        self.co2 = 400.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.temp, self.outdoor_temp, self.occupancy, self.t, self.co2], dtype=np.float32)

    def step(self, action):
        assert action in (0, 1), "Invalid action: must be 0 (OFF) or 1 (ON)"
        self.temp = self.model.step(self.temp, self.outdoor_temp, action, self.occupancy, self.dt)
        self.t += 1
        done = self.t >= self.max_steps

        if not done:
            self.outdoor_temp = float(self._outdoor_temp_series[self.t])
            self.occupancy = int(self._occupancy_series[self.t])

        energy_penalty = -1.0 * action
        if self.occupancy == 1:
            comfort_penalty = -np.clip(abs(self.temp - 24.0) / 5.0, 0, 2.0)
        else:
            comfort_penalty = -np.clip(abs(self.temp - 24.0) / 15.0, 0, 0.3)
        carbon_penalty = -0.3 * action

        reward = (0.4 * energy_penalty) + (0.4 * comfort_penalty) + (0.2 * carbon_penalty)
        return self._get_obs(), reward, done, False, {
            "energy": energy_penalty, "comfort": comfort_penalty, "carbon": carbon_penalty
        }
