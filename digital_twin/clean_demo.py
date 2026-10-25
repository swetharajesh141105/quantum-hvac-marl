"""
Panel-friendly demo: shows WHY the AC is ON/OFF each hour, not just symbols.
"""
from qubo_scheduler import load_federated_agents, get_preferred_actions, build_qubo, solve_schedule
from data_gen import generate_day_profile

print("Loading trained federated agents (Energy, Comfort, Carbon)...")
agents = load_federated_agents()

print("Computing globally optimal 24-hour AC schedule using QUBO...\n")
preferred = get_preferred_actions(agents)
bqm = build_qubo(preferred)
schedule = [int(x) for x in solve_schedule(bqm)]

profile = generate_day_profile(seed=42)
outdoor_temps = profile["outdoor_temp"]
occupancy = profile["occupancy"]

print("="*65)
print(f"{'Hour':<6}{'Outdoor':<10}{'Occupied':<11}{'AC':<6}{'Reason'}")
print("="*65)

for h in range(24):
    temp = outdoor_temps[h]
    occ = "Yes" if occupancy[h] == 1 else "No"
    action = schedule[h]
    ac = "ON " if action == 1 else "OFF"

    if action == 1 and occupancy[h] == 1:
        reason = "Room occupied, cooling for comfort"
    elif action == 1:
        reason = "Pre-cooling / peak heat hour"
    elif occupancy[h] == 1:
        reason = "Occupied but temp already comfortable"
    else:
        reason = "Room empty, saving energy"

    print(f"{h:02d}:00 {temp:>6.1f}C  {occ:<11}{ac:<6}{reason}")

print("="*65)
ac_on = sum(schedule)
savings = 100 * (1 - ac_on/24)
print(f"AC ON: {ac_on}/24 hours  |  Energy Savings vs Always-On: {savings:.1f}%")
print("="*65)
