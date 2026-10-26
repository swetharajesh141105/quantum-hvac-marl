"""
tradeoff_sweep.py — Change 3: compare against a whole FRONTIER, not one point.

A thermostat baseline is swept over set-points (22-29 C): low set-point = comfy
but expensive, high set-point = cheap but uncomfortable. MARL and
MARL+QUBO-planner are plotted as points on the same cost-vs-discomfort plane;
a point left/below the thermostat curve is a genuine win.

Run: python tradeoff_sweep.py   -> tradeoff_curve.png + tradeoff_curve.csv
"""
import numpy as np
import pandas as pd
from tou_env import rollout_policy, evaluate_schedule
from day_ahead_planner import TEST_SEEDS, get_prefs_fn, tune, plan

SETPOINTS = np.arange(22.0, 29.5, 0.5)


def thermostat_policy(setpoint):
    """ON only when occupied and room is warmer than the set-point."""
    return lambda obs: int(obs[2] == 1 and obs[0] > setpoint)


def mean_metrics(ms):
    return {k: float(np.mean([m[k] for m in ms])) for k in ("cost", "carbon", "violation_hours")}


def run():
    rows = []
    for sp in SETPOINTS:
        m = mean_metrics([rollout_policy(thermostat_policy(sp), s)[1] for s in TEST_SEEDS])
        rows.append({"method": f"thermostat {sp:.1f}C", "kind": "thermostat", **m})

    fn, label = get_prefs_fn()
    w = tune(fn)
    m = mean_metrics([evaluate_schedule(fn(s), s) for s in TEST_SEEDS])
    rows.append({"method": f"MARL ({label})", "kind": "marl", **m})
    m = mean_metrics([evaluate_schedule(plan(fn(s), **w), s) for s in TEST_SEEDS])
    rows.append({"method": "MARL + QUBO planner", "kind": "qubo", **m})

    df = pd.DataFrame(rows)
    df.to_csv("tradeoff_curve.csv", index=False)
    print(df.round(2).to_string(index=False))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = df[df.kind == "thermostat"]
    plt.figure(figsize=(7, 5))
    plt.plot(th.cost, th.violation_hours, "o-", color="gray", label="Thermostat set-point sweep")
    for kind, mk, c in [("marl", "s", "tab:blue"), ("qubo", "*", "tab:red")]:
        r = df[df.kind == kind].iloc[0]
        plt.scatter(r.cost, r.violation_hours, marker=mk, s=160, color=c, label=r.method, zorder=3)
    plt.xlabel("Energy cost per day (tariff-weighted ON-hours)")
    plt.ylabel("Comfort violation hours per day")
    plt.title(f"Cost vs comfort trade-off (test seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("tradeoff_curve.png", dpi=150)
    print("Saved tradeoff_curve.png / tradeoff_curve.csv")


if __name__ == "__main__":
    run()
