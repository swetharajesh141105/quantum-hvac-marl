"""
tou_agents.py — trains/loads the 3 PPO agents (energy, comfort, carbon) on the
tariff-aware env from tou_env.py and checks they did NOT collapse to always-OFF.

Run:  python tou_agents.py            (from the digital_twin folder)
Saves: energy_tou.zip, comfort_tou.zip, carbon_tou.zip  (old zips untouched)
"""
import os, sys, zipfile, tempfile
import numpy as np
from tou_env import (TariffHVACEnv, TariffWrapper, rollout_policy, HOURS,
                     COMFORT_TARGET, COMFORT_TOL)

OBJECTIVES = ["energy", "comfort", "carbon"]
TRAIN_SEEDS = list(range(200, 220))        # never used for tuning or testing


class _SeedCycler(TariffWrapper):
    """Different outdoor-temp day each episode so agents don't memorise one day."""
    def __init__(self, env, objective):
        super().__init__(env, objective)
        self._i = 0

    def reset(self, **kw):
        seed = TRAIN_SEEDS[self._i % len(TRAIN_SEEDS)]
        self._i += 1
        return self.env.reset(seed=seed)


def train_all(timesteps=60000, seed=42):
    from stable_baselines3 import PPO
    for obj in OBJECTIVES:
        print(f"Training {obj} agent on tariff-aware env...")
        env = _SeedCycler(TariffHVACEnv(), obj)
        model = PPO("MlpPolicy", env, verbose=0, ent_coef=0.02, seed=seed)
        model.learn(total_timesteps=timesteps)
        model.save(f"{obj}_tou")


def load_tou_agent(objective):
    """Same Windows-safe loader idea as marl_agents.robust_load, 7-dim obs."""
    import torch
    from stable_baselines3 import PPO
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(f"{objective}_tou.zip") as z:
        z.extractall(tmp)
    model = PPO("MlpPolicy", TariffWrapper(TariffHVACEnv(), objective), verbose=0)
    model.policy.load_state_dict(
        torch.load(os.path.join(tmp, "policy.pth"), map_location="cpu", weights_only=True))
    return model


def joint_action(obs, agents):
    """Weighted vote (comfort 2x) + hard override: occupied & too hot -> ON."""
    votes = {k: int(a.predict(obs, deterministic=True)[0]) for k, a in agents.items()}
    action = int(2 * votes["comfort"] + votes["energy"] + votes["carbon"] >= 2)
    temp, occ = obs[0], obs[2]
    if occ == 1 and temp > COMFORT_TARGET + COMFORT_TOL:
        action = 1
    elif temp < COMFORT_TARGET - COMFORT_TOL:
        action = 0
    return action


def collapse_report(agents, seeds=range(0, 5)):
    """Per-agent fraction of hours ON (solo) — ~0.0 means it collapsed to OFF."""
    print("Collapse check (fraction of hours each agent votes ON):")
    ok = True
    for name, ag in agents.items():
        fr = np.mean([rollout_policy(lambda o: ag.predict(o, deterministic=True)[0], s)[0].mean()
                      for s in seeds])
        print(f"  {name:8s}: {fr:.2f}")
        ok &= fr > 0.05
    return ok


if __name__ == "__main__":
    train_all()
    agents = {o: load_tou_agent(o) for o in OBJECTIVES}
    print("No collapse." if collapse_report(agents) else "WARNING: an agent still collapsed to OFF.")
    _, m = rollout_policy(lambda o: joint_action(o, agents), seed=0)
    print("Joint policy, seed 0:", m)
