"""
visualize.py — Person C (Harshavardhini)

Runs HVACEnv (Person B) for one simulated day using fixed/random actions,
and plots temperature, AC state, and occupancy over time.

Purpose: visually sanity-check that A + B's work behaves realistically
before anyone starts training RL agents on top of it.

--- INTEGRATION NOTE (flag to Swetha / Dharshinikesan) ---
HVACEnv currently generates its own outdoor_temp/occupancy internally via
_get_current_conditions() rather than consuming data_gen.py's
generate_day_profile(). This script therefore does NOT feed data_gen.py's
output into the env yet -- it just runs the env as-is and records what it
reports via `obs`. Once the team wires data_gen.py into HVACEnv, this
script can be updated to pass that profile in and remove the duplication.

obs layout confirmed from hvac_env.py: [room_temp, outdoor_temp, occupancy, hour_of_day, co2]
step() returns an empty info dict -- everything relevant is in obs.
dt = 0.25 (hours, not seconds); max_steps = 96 -> 24h simulated day.
"""

import matplotlib.pyplot as plt
import numpy as np

from hvac_env import HVACEnv

# obs vector indices, per hvac_env.py's _get_obs()
IDX_ROOM_TEMP = 0
IDX_OUTDOOR_TEMP = 1
IDX_OCCUPANCY = 2
IDX_HOUR = 3
IDX_CO2 = 4


def run_one_day(env, policy: str = "random", seed: int = 42):
    """
    Steps through the environment for a full day and records history.

    policy:
        "random" -> env.action_space.sample() each step
        "off"    -> AC always off (action 0) -- confirms temp rises
                    realistically with no cooling
        "on"     -> AC always on (action 1) -- confirms temp falls
                    realistically with full cooling
    """
    history = {"room_temp": [], "outdoor_temp": [], "ac_state": [], "occupancy": []}

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "off":
            action = 0
        elif policy == "on":
            action = 1
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, terminated, truncated, info = env.step(action)

        history["room_temp"].append(obs[IDX_ROOM_TEMP])
        history["outdoor_temp"].append(obs[IDX_OUTDOOR_TEMP])
        history["occupancy"].append(obs[IDX_OCCUPANCY])
        history["ac_state"].append(action)

    return {k: np.array(v) for k, v in history.items()}


def plot_day(history: dict, dt_hours: float, title: str = "Digital Twin -- One Simulated Day"):
    n_steps = len(history["room_temp"])
    t_hours = np.arange(n_steps) * dt_hours

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t_hours, history["room_temp"], label="Room temp (C)", color="tab:blue")
    axes[0].plot(t_hours, history["outdoor_temp"], label="Outdoor temp (C)", color="tab:orange", alpha=0.6)
    axes[0].axhline(20.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Comfort band")
    axes[0].axhline(25.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].set_ylabel("Temp (C)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].step(t_hours, history["ac_state"], where="post", color="tab:green")
    axes[1].set_ylabel("AC state")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["OFF", "ON"])
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(t_hours, history["occupancy"], step="post", color="tab:purple", alpha=0.5)
    axes[2].set_ylabel("Occupancy")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Empty", "Occupied"])
    axes[2].set_xlabel("Hour of day")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    env = HVACEnv()
    dt_hours = env.dt

    for policy in ["off", "on", "random"]:
        print(f"Running policy: {policy}")
        history = run_one_day(env, policy=policy, seed=42)
        fig = plot_day(history, dt_hours=dt_hours, title=f"Digital Twin -- One Day (policy: {policy})")
        out_path = f"day_simulation_{policy}.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved {out_path}  |  temp range: {history['room_temp'].min():.1f}-{history['room_temp'].max():.1f}C")

    plt.show()
