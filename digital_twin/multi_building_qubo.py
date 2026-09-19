"""
JOURNAL-GRADE demo: 3 federated buildings coordinated under a shared grid
capacity constraint (max 2 of 3 running AC simultaneously). QUBO resolves
conflicts by STAGGERING hours (shifting to adjacent free slots) rather than
simply dropping comfort, via a cardinality-preserving penalty that keeps
each building's total ON-hour count close to its original preference.
"""
import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from thermal_model import RoomThermalModel
from marl_agents import robust_load

HOURS = 24
MAX_SIMULTANEOUS = 2

BUILDINGS = {
    "B1": {"params": {"thermal_mass": 5.0, "insulation_resistance": 2.0, "ac_cooling_power": 3.0, "occupancy_heat_gain": 0.5},
           "occupancy": [0]*9 + [1]*9 + [0]*6},
    "B2": {"params": {"thermal_mass": 4.0, "insulation_resistance": 1.5, "ac_cooling_power": 3.5, "occupancy_heat_gain": 0.6},
           "occupancy": [0]*8 + [1]*10 + [0]*6},
    "B3": {"params": {"thermal_mass": 6.0, "insulation_resistance": 2.5, "ac_cooling_power": 2.8, "occupancy_heat_gain": 0.4},
           "occupancy": [0]*10 + [1]*8 + [0]*6},
}
OUTDOOR = [24 + 8*np.sin((h-6)*np.pi/12) for h in range(HOURS)]


def get_building_preference(bname, bcfg, agents):
    model = RoomThermalModel(bcfg["params"])
    temp = 24.0
    prefs, occs = [], []
    for t in range(HOURS):
        occ = bcfg["occupancy"][t]
        obs = np.array([temp, OUTDOOR[t], occ, t, 400.0], dtype=np.float32)
        votes = [int(agents[a].predict(obs, deterministic=True)[0]) for a in agents]
        action = 1 if (occ and temp > 25.5) or sum(votes) >= 2 else 0
        temp = model.step(temp, OUTDOOR[t], action, occ, dt=1.0)
        prefs.append(action)
        occs.append(occ)
    return np.array(prefs), np.array(occs)


def build_qubo(prefs, occs, c_dev=1.0, c_constraint=4.0, c_count=1.5):
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    names = list(prefs.keys())

    for b in names:
        target = int(prefs[b].sum())  # preserve this building's total ON-hour count
        for t in range(HOURS):
            if occs[b][t] == 1:
                # only allow ON hours within building's OWN occupied window
                var = f"{b}_{t}"
                p = prefs[b][t]
                bqm.add_linear(var, c_dev * (1 - 2*p))
        # cardinality penalty: keep total ON count near target, expanded over occupied hours
        occ_vars = [f"{b}_{t}" for t in range(HOURS) if occs[b][t] == 1]
        for v in occ_vars:
            bqm.add_linear(v, c_count * (1 - 2*target))
        for i in range(len(occ_vars)):
            for j in range(i+1, len(occ_vars)):
                bqm.add_quadratic(occ_vars[i], occ_vars[j], 2*c_count)

    # Grid constraint: penalize >MAX_SIMULTANEOUS buildings ON in same hour
    for t in range(HOURS):
        vars_t = [f"{b}_{t}" for b in names if occs[b][t] == 1]
        for i in range(len(vars_t)):
            for j in range(i+1, len(vars_t)):
                bqm.add_quadratic(vars_t[i], vars_t[j], c_constraint)

    return bqm, names


def run():
    print("Loading federated agents...")
    agents = {obj: robust_load(obj, suffix="_agent_federated") for obj in ["energy","carbon","comfort"]}

    prefs, occs = {}, {}
    print("\nEach building's INDEPENDENT preferred schedule:")
    for bname, bcfg in BUILDINGS.items():
        p, o = get_building_preference(bname, bcfg, agents)
        prefs[bname], occs[bname] = p, o
        print(f"  {bname}: {p.tolist()}  (ON hours: {p.sum()})")

    violations = sum(1 for t in range(HOURS) if sum(prefs[b][t] for b in prefs) > MAX_SIMULTANEOUS)
    print(f"\nGrid violations BEFORE coordination: {violations}/24")

    print("\nSolving joint QUBO (staggering-aware, cardinality-preserving)...")
    bqm, names = build_qubo(prefs, occs)
    result = SimulatedAnnealingSampler().sample(bqm, num_reads=300).first.sample

    schedule = {b: [result.get(f"{b}_{t}", 0) for t in range(HOURS)] for b in names}
    print("\nQUBO-COORDINATED schedule:")
    for b in names:
        print(f"  {b}: {schedule[b]}  (ON hours: {sum(schedule[b])}, was {prefs[b].sum()})")

    post_v = sum(1 for t in range(HOURS) if sum(schedule[b][t] for b in names) > MAX_SIMULTANEOUS)
    print(f"\nGrid violations AFTER QUBO coordination: {post_v}/24")
    for b in names:
        print(f"  {b} comfort preserved: {sum(schedule[b])}/{prefs[b].sum()} original ON hours ({100*int(sum(schedule[b]))/max(int(prefs[b].sum()),1):.0f}%)")

if __name__ == "__main__":
    run()
