import numpy as np, dimod
from dwave.samplers import SimulatedAnnealingSampler
from multi_building_qubo import *
def build(prefs, occs, c_dev=1.0, c_count=3.0, c_grid=10.0, M=6.0):
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY); names = list(prefs)
    for b in names:
        K = int(prefs[b].sum()); vs = [f"{b}_{t}" for t in range(HOURS) if occs[b][t]]
        for t in range(HOURS):
            if occs[b][t]: bqm.add_linear(f"{b}_{t}", c_dev*(1-2*int(prefs[b][t])))
        for v in vs: bqm.add_linear(v, c_count*(1-2*K))
        for i in range(len(vs)):
            for j in range(i+1, len(vs)): bqm.add_quadratic(vs[i], vs[j], 2*c_count)
    for t in range(HOURS):
        x = [f"{b}_{t}" for b in names if occs[b][t]]
        if len(x) == 3:
            z = f"z_{t}"
            bqm.add_quadratic(x[0], x[1], M); bqm.add_quadratic(x[0], z, -2*M)
            bqm.add_quadratic(x[1], z, -2*M); bqm.add_linear(z, 3*M)
            bqm.add_quadratic(z, x[2], c_grid)
    return bqm
def viol(bname, sched):
    cfg = BUILDINGS[bname]; m = RoomThermalModel(cfg["params"]); T = 24.0; v = 0
    for t in range(HOURS):
        T = m.step(T, OUTDOOR[t], int(sched[t]), cfg["occupancy"][t], 1.0)
        v += int(cfg["occupancy"][t] and abs(T-24.0) > 2.0)
    return v
agents = {a: robust_load(a, suffix="_agent_federated") for a in ["energy", "carbon", "comfort"]}
prefs, occs = {}, {}
for b, cfg in BUILDINGS.items(): prefs[b], occs[b] = get_building_preference(b, cfg, agents)
res = SimulatedAnnealingSampler().sample(build(prefs, occs), num_reads=500, seed=1).first.sample
sched = {b: [int(res.get(f"{b}_{t}", 0)) for t in range(HOURS)] for b in prefs}
gv = lambda S: sum(1 for t in range(HOURS) if sum(int(S[b][t]) for b in S) > MAX_SIMULTANEOUS)
print(f"Grid violations: before {gv(prefs)}/24, after {gv(sched)}/24")
for b in prefs:
    print(f" {b}: ON hours {int(prefs[b].sum())} -> {sum(sched[b])} | room comfort violations {viol(b, prefs[b])} -> {viol(b, sched[b])} h")
