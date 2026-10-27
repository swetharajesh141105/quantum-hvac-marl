import numpy as np, dimod
from dwave.samplers import SimulatedAnnealingSampler
from tariff_qubo_demo import *
a = PARAMS["ac_cooling_power"]/PARAMS["thermal_mass"]
d = 1 - 1/(PARAMS["insulation_resistance"]*PARAMS["thermal_mass"])
pn = PRICE/PRICE.max()
def replan(k, T0, fc, lam, target=24.5, c_sw=0.05):
    m = RoomThermalModel(PARAMS); T = T0; Tf = {}
    for t in range(k, H):
        T = m.step(T, fc[t], 0, OCC[t], 1.0); Tf[t] = T
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for t in range(k, H): bqm.add_variable(t, lam*pn[t] if t in WINDOW else 50.0)
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
def closed_loop(true, fc, lam):
    m = RoomThermalModel(PARAMS); T = 24.0; s = []; v = 0
    for k in range(H):
        x = replan(k, T, fc, lam) if k in WINDOW else 0
        T = m.step(T, true[k], x, OCC[k], 1.0); s.append(x)
        v += int(OCC[k] and abs(T-24.0) > 2.0)
    return cost(s), v
N, SG, LAM = 20, 2.0, 1.0
R = {"Thermostat": [], "Open-loop QUBO": [], "Closed-loop QUBO": []}
for i in range(N):
    r = np.random.RandomState(i)
    true = [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
    fc = list(np.array(true)+np.random.RandomState(100+i).normal(0,SG,H))
    b = thermostat(true); R["Thermostat"].append((cost(b), simulate(b,true)))
    o = qubo_plan(fc, LAM); R["Open-loop QUBO"].append((cost(o), simulate(o,true)))
    R["Closed-loop QUBO"].append(closed_loop(true, fc, LAM))
print(f"Forecast error {SG}C, {N} days")
for k, v in R.items():
    v = np.array(v); print(f" {k:18s} cost Rs {v[:,0].mean():5.1f}/day  violations {v[:,1].mean():.2f} h/day")
