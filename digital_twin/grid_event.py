import numpy as np, dimod
from dwave.samplers import SimulatedAnnealingSampler
from tariff_qubo_demo import *
a = PARAMS["ac_cooling_power"]/PARAMS["thermal_mass"]
d = 1 - 1/(PARAMS["insulation_resistance"]*PARAMS["thermal_mass"])
pn = PRICE/PRICE.max(); BLK = {14, 15, 16}; NOTICE = 10; LAM = 1.0
def replan(k, T0, out, blk, target=24.5, c_sw=0.05):
    m = RoomThermalModel(PARAMS); T = T0; Tf = {}
    for t in range(k, H):
        T = m.step(T, out[t], 0, OCC[t], 1.0); Tf[t] = T
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for t in range(k, H):
        bqm.add_variable(t, 1000.0 if t in blk else (LAM*pn[t] if t in WINDOW else 50.0))
    for s in range(k, H):
        if not OCC[s]: continue
        r = Tf[s]-target
        co = {t: a*d**(s-t) for t in WINDOW if k <= t <= s}
        for t, c in co.items(): bqm.add_linear(t, -2*r*c + c*c)
        ts = list(co)
        for i in range(len(ts)):
            for j in range(i+1, len(ts)): bqm.add_quadratic(ts[i], ts[j], 2*co[ts[i]]*co[ts[j]])
    for t in range(k, H-1):
        bqm.add_linear(t, c_sw); bqm.add_linear(t+1, c_sw); bqm.add_quadratic(t, t+1, -2*c_sw)
    return int(SimulatedAnnealingSampler().sample(bqm, num_reads=100, seed=1).first.sample[k])
def run(policy, out):
    m = RoomThermalModel(PARAMS); T = 24.0; s = []; v = 0
    for k in range(H):
        x = policy(k, T, out)
        if k in BLK: x = 0
        T = m.step(T, out[k], x, OCC[k], 1.0); s.append(x)
        v += int(OCC[k] and abs(T-24.0) > 2.0)
    return cost(s), v
def p_thermo(k, T, out): return int(OCC[k] and T > 25.0)
def p_closed(k, T, out):
    if k not in WINDOW: return 0
    return replan(k, T, out, BLK if k >= NOTICE else set())
R = {"Thermostat": [], "Morning plan (no re-plan)": [], "Closed-loop QUBO": []}
for i in range(20):
    r = np.random.RandomState(i)
    out = [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
    plan = qubo_plan(out, LAM)
    R["Thermostat"].append(run(p_thermo, out))
    R["Morning plan (no re-plan)"].append(run(lambda k,T,o: int(plan[k]), out))
    R["Closed-loop QUBO"].append(run(p_closed, out))
print("Grid event: AC banned 14:00-17:00, announced at 10:00 (20 days)")
for k, v in R.items():
    v = np.array(v); print(f" {k:26s} cost Rs {v[:,0].mean():5.1f}/day  violations {v[:,1].mean():.2f} h/day")
