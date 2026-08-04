"""
data_gen.py — Person C (Harshavardhini)

Generates realistic synthetic inputs consumed by HVACEnv (Person B) and,
indirectly, RoomThermalModel (Person A). Owned solely by Person C —
A and B should only ever IMPORT from this file, never edit it.

Fixed contract (per team plan): generate_day_profile(seed) -> dict of arrays.
"""

import os
import numpy as np
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load the single source of truth for all tunable constants."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_outdoor_temp_curve(hours: int, timestep_seconds: int, cfg: dict, seed: int) -> np.ndarray:
    """
    24h (or `hours`-long) synthetic outdoor temperature curve:
    sine wave (day/night cycle) + gaussian noise.
    Returns an array of length (hours * 3600 / timestep_seconds).
    """
    rng = np.random.default_rng(seed)
    steps_per_hour = 3600 / timestep_seconds
    n_steps = int(hours * steps_per_hour)

    t_hours = np.linspace(0, hours, n_steps, endpoint=False)

    base = cfg["outdoor_temp"]["base_temp_c"]
    amp = cfg["outdoor_temp"]["amplitude_c"]
    peak_hour = cfg["outdoor_temp"]["peak_hour"]
    noise_std = cfg["outdoor_temp"]["noise_std_c"]

    # Sine wave peaking at `peak_hour` (shift so cos peaks at peak_hour)
    curve = base + amp * np.cos((2 * np.pi / 24) * (t_hours - peak_hour))
    noise = rng.normal(0, noise_std, size=n_steps)
    return curve + noise


def generate_occupancy_schedule(hours: int, timestep_seconds: int, cfg: dict, seed: int) -> np.ndarray:
    """
    Binary occupancy schedule (1 = occupied, 0 = empty), time-based pattern
    e.g. occupied from start_hour to end_hour, with small random edge noise
    so it's not a perfectly sharp step function (more realistic).
    """
    rng = np.random.default_rng(seed + 1)  # different stream than temp noise
    steps_per_hour = 3600 / timestep_seconds
    n_steps = int(hours * steps_per_hour)

    t_hours = np.linspace(0, hours, n_steps, endpoint=False)
    start = cfg["occupancy"]["start_hour"]
    end = cfg["occupancy"]["end_hour"]

    occupancy = ((t_hours >= start) & (t_hours < end)).astype(int)

    # Flip ~1% of steps near the boundaries to simulate early arrivals / late leavers
    flip_mask = rng.random(n_steps) < 0.01
    occupancy = np.where(flip_mask, 1 - occupancy, occupancy)
    return occupancy


def generate_co2_placeholder(occupancy: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Placeholder CO2 curve — baseline + increment while occupied.
    Not physically simulated; just enough for HVACEnv's observation_space
    to have a real-looking placeholder value until a proper model exists.
    """
    baseline = cfg["co2"]["baseline_ppm"]
    bump = cfg["co2"]["occupied_increment_ppm"]
    return baseline + occupancy * bump


def generate_day_profile(seed: int = None, cfg: dict = None) -> dict:
    """
    Fixed contract: returns a dict of arrays consumed by HVACEnv.

    Keys:
        outdoor_temp : np.ndarray, shape (n_steps,)
        occupancy    : np.ndarray, shape (n_steps,)  (0/1)
        co2          : np.ndarray, shape (n_steps,)  (placeholder)
        timestamps   : np.ndarray, shape (n_steps,)  (hours, 0..duration)
        dt           : float, seconds per step
    """
    if cfg is None:
        cfg = load_config()
    if seed is None:
        seed = cfg["simulation"]["seed"]

    hours = cfg["simulation"]["duration_hours"]
    dt = cfg["simulation"]["timestep_seconds"]

    outdoor_temp = generate_outdoor_temp_curve(hours, dt, cfg, seed)
    occupancy = generate_occupancy_schedule(hours, dt, cfg, seed)
    co2 = generate_co2_placeholder(occupancy, cfg)
    n_steps = len(outdoor_temp)
    timestamps = np.linspace(0, hours, n_steps, endpoint=False)

    return {
        "outdoor_temp": outdoor_temp,
        "occupancy": occupancy,
        "co2": co2,
        "timestamps": timestamps,
        "dt": dt,
    }


if __name__ == "__main__":
    # Quick sanity check when run standalone
    cfg = load_config()
    profile = generate_day_profile(cfg=cfg)

    print("Generated day profile:")
    for key, val in profile.items():
        if hasattr(val, "shape"):
            print(f"  {key:12s} shape={val.shape}  min={val.min():.2f}  max={val.max():.2f}")
        else:
            print(f"  {key:12s} = {val}")

    occ_hours = profile["occupancy"].sum() * (profile["dt"] / 3600)
    print(f"\nTotal occupied hours: {occ_hours:.1f}h")
    print(f"Outdoor temp range: {profile['outdoor_temp'].min():.1f}C to {profile['outdoor_temp'].max():.1f}C")
