import numpy as np
from thermal_model import RoomThermalModel
from marl_agents import robust_load, ROOM_PARAMS
from qubo_scheduler import build_qubo, solve_schedule
from xai_explainer import explain_schedule

HOURS = 24
OCCUPANCY = [0]*9 + [1]*4 + [0] + [1]*4 + [0]*6
OUTDOOR = [24 + 8*np.sin((h-6)*np.pi/12) for h in range(HOURS)]
COMFORT_TEMP, TOLERANCE = 24.0, 1.5

def precool_signal(model, temp, t):
    gap_end = next((i for i in range(t+1, HOURS) if OCCUPANCY[i] == 1), None)
    if gap_end == t + 1:
        projected = model.step(temp, OUTDOOR[t], 0, 0, dt=1.0)
        return int(abs(projected - COMFORT_TEMP) > TOLERANCE)
    return 0

def decide(temp, occupied, energy_vote, carbon_vote, comfort_vote=None):
    if occupied:
        if temp > COMFORT_TEMP + TOLERANCE:
            return 1, "Occupied, too warm -> AC ON for comfort."
        if temp < COMFORT_TEMP - TOLERANCE:
            return 0, "Occupied, already cool -> AC OFF."
        votes = [energy_vote, carbon_vote] + ([comfort_vote] if comfort_vote is not None else [])
        action = 1 if sum(votes) >= len(votes)/2 else 0
        return action, f"Occupied, borderline -> agents decided {'ON' if action else 'OFF'}."
    return 0, "Room empty -> AC OFF."

def run_pipeline():
    print("=== STEP 1-3: Loading Digital Twin + federated MARL agents ===")
    model = RoomThermalModel(ROOM_PARAMS)
    agents = {obj: robust_load(obj) for obj in ["energy", "carbon", "comfort"]}
    print("\n=== STEP 4: Deciding hour-by-hour ===")
    temp = COMFORT_TEMP
    actions = []
    for t in range(HOURS):
        occ_signal = OCCUPANCY[t] or precool_signal(model, temp, t)
        obs = np.array([temp, OUTDOOR[t], occ_signal, t, 400.0], dtype=np.float32)
        e_vote = int(agents["energy"].predict(obs, deterministic=True)[0])
        c_vote = int(agents["carbon"].predict(obs, deterministic=True)[0])
        comfort_vote = int(agents["comfort"].predict(obs, deterministic=True)[0])
        action, reason = decide(temp, occ_signal, e_vote, c_vote, comfort_vote)
        temp = model.step(temp, OUTDOOR[t], action, OCCUPANCY[t], dt=1.0)
        actions.append(action)
        print(f"Hour {t:02d}: {reason} (temp={temp:.1f}C)")
    preferred = np.array(actions)
    print("\n=== STEP 5: QUBO ===")
    bqm = build_qubo(preferred, c_energy=0.3, c_deviation=0.9, c_switch=0.3)
    schedule = solve_schedule(bqm)
    print("\n=== STEP 6: XAI ===")
    print(explain_schedule(schedule, preferred))
    marl_savings = 100 * (1 - sum(preferred)/HOURS)
    qubo_savings = 100 * (1 - sum(schedule)/HOURS)
    changed = sum(1 for a, b in zip(preferred, schedule) if a != b)
    print(f"\n=== SUMMARY ===\nComfort-first savings: {marl_savings:.1f}%\nQUBO-smoothed savings: {qubo_savings:.1f}%\nHours QUBO changed: {changed}/24")

if __name__ == "__main__":
    run_pipeline()
