"""
Demonstrates QUBO's real value: with noisy sensor readings (realistic —
real DHT22/PIR sensors have jitter), agents can flicker ON/OFF rapidly.
QUBO smooths this into a stable, switch-minimized schedule.
"""
import numpy as np
from thermal_model import RoomThermalModel
from marl_agents import robust_load, ROOM_PARAMS
from qubo_scheduler import build_qubo, solve_schedule
from xai_explainer import explain_schedule

HOURS = 24
OCCUPANCY = [0]*9 + [1]*4 + [0] + [1]*4 + [0]*6
OUTDOOR = [24 + 8*np.sin((h-6)*np.pi/12) for h in range(HOURS)]
np.random.seed(3)  # reproducible noise

def run():
    model = RoomThermalModel(ROOM_PARAMS)
    agents = {obj: robust_load(obj, suffix="_agent_federated") for obj in ["energy","carbon","comfort"]}
    temp = 24.0
    actions = []
    for t in range(HOURS):
        noisy_temp = temp + np.random.normal(0, 0.8)  # realistic sensor jitter
        occ = OCCUPANCY[t]
        obs = np.array([noisy_temp, OUTDOOR[t], occ, t, 400.0], dtype=np.float32)
        votes = [int(agents[a].predict(obs, deterministic=True)[0]) for a in agents]
        action = 1 if sum(votes) >= 2 else 0
        temp = model.step(temp, OUTDOOR[t], action, occ, dt=1.0)
        actions.append(action)
        print(f"Hour {t:02d}: sensor={noisy_temp:.1f}C real={temp:.1f}C occ={occ} -> agents vote {'ON' if action else 'OFF'}")

    preferred = np.array(actions)
    bqm = build_qubo(preferred, c_energy=0.3, c_deviation=0.5, c_switch=0.6)
    schedule = solve_schedule(bqm)
    print("\n" + explain_schedule(schedule, preferred))
    changed = sum(1 for a,b in zip(preferred, schedule) if a != b)
    print(f"\nAgent switches: {sum(1 for i in range(1,24) if preferred[i]!=preferred[i-1])}")
    print(f"QUBO switches:  {sum(1 for i in range(1,24) if schedule[i]!=schedule[i-1])}")
    print(f"Hours QUBO changed: {changed}/24")

if __name__ == "__main__":
    run()
