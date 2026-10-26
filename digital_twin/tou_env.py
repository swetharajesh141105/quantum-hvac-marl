"""
tou_env.py — Change 1: fix "agent collapse" (all agents learning always-OFF).

Why the old env collapses: ON costs energy/carbon every hour, comfort is only
a small weighted term, and nothing is shared between agents, so the energy and
carbon agents find that OFF is always the best reward.

What changes here:
  * the agents SEE a time-of-use tariff and a grid carbon-intensity signal
    (obs grows from 5 to 7 values: [..old 5.., tariff, carbon_intensity]),
  * ON now costs tariff[t] (not a flat 1.0), so cheap hours are worth using,
  * a SHARED comfort-violation penalty is subtracted from EVERY agent's reward,
    so "always OFF" is no longer optimal for any of them.

NOTE: the observation shape changes (5 -> 7), so the existing *_agent*.zip
checkpoints cannot be loaded here. Train new ones with tou_agents.py; the
old files are not touched.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from hvac_env import HVACEnv

HOURS = 24
COMFORT_TARGET, COMFORT_TOL = 24.0, 1.5      # same band as run_full_pipeline.py

# Rs/kWh-style relative price: night cheap, afternoon peak (overlaps occupancy)
TARIFF = np.array([0.5]*6 + [1.0]*8 + [2.0]*4 + [1.0]*4 + [0.5]*2, dtype=np.float32)
# relative grid carbon intensity: solar dip midday, evening peak
CARBON = np.array([0.7]*6 + [0.6]*4 + [0.4]*6 + [0.8]*2 + [1.0]*4 + [0.7]*2, dtype=np.float32)
assert len(TARIFF) == HOURS and len(CARBON) == HOURS

# per-agent weights on (cost, comfort, carbon); the shared penalty is on top
TOU_WEIGHTS = {
    "energy":  (0.6, 0.2, 0.2),
    "comfort": (0.1, 0.8, 0.1),
    "carbon":  (0.2, 0.2, 0.6),
}
SHARED_VIOLATION_PENALTY = 2.0   # identical for all three agents


class TariffHVACEnv(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=0, high=2000, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_obs(), {}

    def _get_obs(self):
        base = super()._get_obs()
        h = min(self.t, HOURS - 1)
        return np.concatenate([base, [TARIFF[h], CARBON[h]]]).astype(np.float32)

    def step(self, action):
        t0, occ0 = self.t, self.occupancy          # hour being acted on
        _, reward, done, trunc, info = super().step(action)
        h = min(t0, HOURS - 1)
        violation = float(occ0 == 1 and abs(self.temp - COMFORT_TARGET) > COMFORT_TOL)
        info = dict(info,
                    cost=float(TARIFF[h] * action),
                    carbon_kg=float(CARBON[h] * action),
                    violation=violation)
        return self._get_obs(), reward, done, trunc, info


class TariffWrapper(gym.Wrapper):
    """Per-agent reward = weighted(cost, comfort, carbon) - SHARED violation penalty."""
    def __init__(self, env, objective):
        super().__init__(env)
        self.objective = objective

    def step(self, action):
        obs, _, done, trunc, info = self.env.step(action)
        we, wc, wcarb = TOU_WEIGHTS[self.objective]
        reward = (we * -(info["cost"] / TARIFF.max())
                  + wc * info["comfort"]
                  + wcarb * -(info["carbon_kg"] / CARBON.max())
                  - SHARED_VIOLATION_PENALTY * info["violation"])
        return obs, float(reward), done, trunc, info


def evaluate_schedule(schedule, seed, room_params=None):
    """Play a fixed 24h ON/OFF schedule through the env for one seed.
    Returns dict: cost, carbon, on_hours, violation_hours."""
    env = TariffHVACEnv(room_params, seed=seed)
    env.reset(seed=seed)
    cost = carbon = viol = 0.0
    for a in schedule:
        _, _, _, _, info = env.step(int(a))
        cost += info["cost"]; carbon += info["carbon_kg"]; viol += info["violation"]
    return {"cost": cost, "carbon": carbon, "on_hours": int(sum(schedule)),
            "violation_hours": viol}


def rollout_policy(policy_fn, seed, room_params=None):
    """Closed-loop rollout: policy_fn(obs) -> 0/1. Returns (actions, metrics)."""
    env = TariffHVACEnv(room_params, seed=seed)
    obs, _ = env.reset(seed=seed)
    acts, cost, carbon, viol = [], 0.0, 0.0, 0.0
    for _ in range(HOURS):
        a = int(policy_fn(obs))
        obs, _, _, _, info = env.step(a)
        acts.append(a); cost += info["cost"]; carbon += info["carbon_kg"]; viol += info["violation"]
    return np.array(acts), {"cost": cost, "carbon": carbon, "on_hours": int(sum(acts)),
                            "violation_hours": viol}
