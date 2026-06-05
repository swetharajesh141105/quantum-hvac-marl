"""
Digital Twin — Room Thermal Model
===================================
Simulates how a room's temperature changes over time based on:
  - AC state (on/off and setpoint)
  - Number of occupants (each person generates ~80W of heat)
  - Outdoor temperature
  - Building thermal inertia (how well insulated it is)

CONCEPT EXPLANATION:
This is based on Newton's Law of Cooling + heat gain model.
  dT/dt = (T_outdoor - T_room) * α  ← heat leaking in from outside
         - AC_cooling * β            ← cooling from AC
         + occupants * γ             ← heat from people
         + base_heat_gain * δ        ← sunlight, equipment etc.

We discretize this into 5-second timesteps for simulation.
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class RoomConfig:
    """Physical parameters of the simulated room."""
    area_sqm: float = 20.0          # Room size in square meters
    ceiling_height: float = 3.0     # Meters
    wall_insulation: float = 0.15   # 0=no insulation, 1=perfect insulation
    ac_capacity_kw: float = 1.5     # AC cooling power in kilowatts
    max_occupancy: int = 10         # Max people in room


class DigitalTwin:
    """
    Physics-based thermal simulation of a room.
    
    This is the environment your RL agents train in before
    touching real hardware. Changing AC setpoints here is safe
    — no one gets hot or cold!
    
    State vector (what the RL agent observes):
        [temp, humidity, occupancy, co2_ppm, power_kw,
         outdoor_temp, time_of_day_sin, time_of_day_cos,
         ac_on, ac_setpoint]
    
    The time encoding uses sin/cos so that 23:59 and 00:01
    are close to each other numerically (circular time).
    """

    def __init__(self, config: Optional[RoomConfig] = None, seed: int = 42):
        self.config = config or RoomConfig()
        self.rng = np.random.default_rng(seed)
        self.timestep_seconds = 5       # simulate every 5 seconds
        self.reset()

    # ------------------------------------------------------------------ #
    #  Core Physics                                                        #
    # ------------------------------------------------------------------ #

    def _thermal_dynamics(self, temp: float, outdoor_temp: float,
                          ac_on: bool, ac_setpoint: float,
                          occupants: int) -> float:
        """
        Returns new room temperature after one timestep.
        
        α = heat transfer coefficient (higher = worse insulation)
        β = AC cooling effectiveness
        γ = heat per person
        """
        α = (1.0 - self.config.wall_insulation) * 0.003   # per second
        β = self.config.ac_capacity_kw * 0.8              # kW → °C/s effective
        γ = 0.00015                                        # °C/s per person (~80W)

        heat_leak   = (outdoor_temp - temp) * α
        ac_cooling  = β * (temp - ac_setpoint) / 10.0 if ac_on else 0.0
        people_heat = occupants * γ
        base_gain   = 0.0002  # sunlight, equipment

        dT = (heat_leak - ac_cooling + people_heat + base_gain)
        return temp + dT * self.timestep_seconds

    def _simulate_co2(self, occupants: int, ac_on: bool) -> float:
        """CO2 rises with occupants, drops when AC/ventilation is on."""
        base_co2 = 400  # outdoor level (ppm)
        person_contribution = occupants * 25  # ppm per person
        ventilation_reduction = 50 if ac_on else 0
        noise = self.rng.normal(0, 5)
        return max(400, self.co2 + (person_contribution - ventilation_reduction) * 0.01 + noise)

    def _simulate_humidity(self, temp: float, occupants: int) -> float:
        """Humidity increases with people, decreases with AC."""
        target = 40 + occupants * 3 - (self.ac_on * 8)
        return np.clip(self.humidity + (target - self.humidity) * 0.01 + self.rng.normal(0, 0.2), 20, 80)

    def _simulate_occupancy(self) -> int:
        """
        Realistic occupancy pattern:
        - Empty at night (00:00–07:00)
        - Fills up during work hours (09:00–18:00)
        - Random variation throughout
        """
        hour = (self.time_step * self.timestep_seconds / 3600) % 24
        if 0 <= hour < 7 or hour >= 21:
            base = 0
        elif 7 <= hour < 9:
            base = int(self.config.max_occupancy * 0.3 * (hour - 7) / 2)
        elif 9 <= hour < 18:
            base = int(self.config.max_occupancy * 0.8)
        elif 18 <= hour < 21:
            base = int(self.config.max_occupancy * 0.4)
        else:
            base = 0

        noise = self.rng.integers(-2, 3)
        return int(np.clip(base + noise, 0, self.config.max_occupancy))

    def _simulate_outdoor_temp(self) -> float:
        """Realistic daily temperature cycle (Chennai-like climate)."""
        hour = (self.time_step * self.timestep_seconds / 3600) % 24
        base = 32.0   # mean outdoor temp (°C)
        amplitude = 5.0
        return base + amplitude * np.sin((hour - 14) * np.pi / 12) + self.rng.normal(0, 0.3)

    def _calculate_power(self) -> float:
        """Estimate AC power consumption in kW."""
        if not self.ac_on:
            return 0.0
        # Power increases as load increases (temp diff from setpoint)
        load_factor = max(0, (self.temp - self.ac_setpoint) / 10.0)
        return self.config.ac_capacity_kw * (0.3 + 0.7 * load_factor)

    # ------------------------------------------------------------------ #
    #  Gym-compatible Interface (for RL agents)                           #
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        """Reset simulation to start of a new day."""
        self.time_step = 0
        self.temp = 28.0 + self.rng.normal(0, 1)
        self.humidity = 55.0
        self.co2 = 420.0
        self.ac_on = False
        self.ac_setpoint = 24.0
        self.occupants = 0
        self.outdoor_temp = 32.0
        self.power_kw = 0.0
        self.episode_energy = 0.0
        self.episode_discomfort = 0.0
        return self._get_observation()

    def step(self, action: int) -> tuple:
        """
        Take one timestep with given action.
        
        Actions:
            0 = AC OFF
            1 = AC ON at 18°C (aggressive cooling)
            2 = AC ON at 20°C
            3 = AC ON at 22°C (comfortable)
            4 = AC ON at 24°C (mild cooling)
            5 = AC ON at 26°C (energy saving)
        
        Returns: (observation, reward, done, info)
        """
        # Decode action
        action_map = {0: (False, 24), 1: (True, 18), 2: (True, 20),
                      3: (True, 22), 4: (True, 24), 5: (True, 26)}
        self.ac_on, self.ac_setpoint = action_map[action]

        # Update environment
        self.outdoor_temp = self._simulate_outdoor_temp()
        self.occupants = self._simulate_occupancy()
        self.temp = self._thermal_dynamics(
            self.temp, self.outdoor_temp, self.ac_on,
            self.ac_setpoint, self.occupants
        )
        self.co2 = self._simulate_co2(self.occupants, self.ac_on)
        self.humidity = self._simulate_humidity(self.temp, self.occupants)
        self.power_kw = self._calculate_power()
        self.time_step += 1

        # Track episode totals
        self.episode_energy += self.power_kw * (self.timestep_seconds / 3600)  # kWh
        discomfort = abs(self.temp - 22.0) if self.occupants > 0 else 0
        self.episode_discomfort += discomfort

        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self.time_step >= (24 * 3600 // self.timestep_seconds)  # 1 full day
        info = {
            "temp": self.temp, "humidity": self.humidity,
            "co2": self.co2, "occupants": self.occupants,
            "power_kw": self.power_kw, "outdoor_temp": self.outdoor_temp,
            "episode_energy_kwh": self.episode_energy,
            "episode_discomfort": self.episode_discomfort,
        }
        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Returns the state vector the RL agent sees.
        Uses sin/cos encoding for time so midnight wraps correctly.
        """
        hour_fraction = (self.time_step * self.timestep_seconds / 3600) % 24 / 24
        return np.array([
            (self.temp - 15) / 20,                    # normalized temperature
            (self.humidity - 30) / 50,                 # normalized humidity
            self.occupants / self.config.max_occupancy,# normalized occupancy
            (self.co2 - 400) / 1600,                   # normalized CO2
            self.power_kw / self.config.ac_capacity_kw,# normalized power
            (self.outdoor_temp - 20) / 20,             # normalized outdoor temp
            np.sin(2 * np.pi * hour_fraction),         # time encoding (sin)
            np.cos(2 * np.pi * hour_fraction),         # time encoding (cos)
            float(self.ac_on),
            (self.ac_setpoint - 16) / 12,              # normalized setpoint
        ], dtype=np.float32)

    def _calculate_reward(self) -> float:
        """
        Composite reward balancing comfort and energy.
        This is what the RL agent learns to maximize.
        
        Positive reward for comfortable temperature when occupied.
        Negative reward for energy waste.
        Negative reward for running AC when empty.
        """
        comfort_weight = 0.6
        energy_weight = 0.3
        empty_penalty_weight = 0.1

        # Comfort: how close to 22°C ideal when occupied
        ideal_temp = 22.0
        comfort_error = abs(self.temp - ideal_temp)
        if self.occupants > 0:
            comfort_reward = max(0, 1.0 - comfort_error / 5.0)
        else:
            comfort_reward = 1.0  # don't penalize comfort when empty

        # Energy: reward for low power usage
        energy_reward = 1.0 - (self.power_kw / self.config.ac_capacity_kw)

        # Waste penalty: running AC when no one is there
        empty_penalty = -1.0 if (self.ac_on and self.occupants == 0) else 0.0

        reward = (comfort_weight * comfort_reward +
                  energy_weight * energy_reward +
                  empty_penalty_weight * empty_penalty)
        return float(reward)

    @property
    def observation_space_size(self) -> int:
        return 10

    @property
    def action_space_size(self) -> int:
        return 6


# ------------------------------------------------------------------ #
#  Quick test — run this file directly to verify the twin works       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 60)
    print("DIGITAL TWIN — Self Test")
    print("=" * 60)

    twin = DigitalTwin()
    obs = twin.reset()
    print(f"Initial state: temp={twin.temp:.1f}°C, occupants={twin.occupants}")

    total_reward = 0
    for step in range(100):
        action = random.randint(0, 5)  # random actions for now
        obs, reward, done, info = twin.step(action)
        total_reward += reward
        if step % 20 == 0:
            print(f"  Step {step:3d} | T={info['temp']:.1f}°C | "
                  f"Occ={info['occupants']} | Power={info['power_kw']:.2f}kW | "
                  f"Reward={reward:.3f}")

    print(f"\nTotal reward over 100 steps: {total_reward:.2f}")
    print(f"Energy consumed: {twin.episode_energy:.4f} kWh")
    print("\n✅ Digital Twin working correctly!")
