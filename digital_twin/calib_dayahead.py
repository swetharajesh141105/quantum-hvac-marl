import numpy as np, tariff_qubo_demo as q
from tariff_qubo_demo import *
TRUE = dict(PARAMS, ac_cooling_power=1.8)
def learn(out):
    m = RoomThermalModel(TRUE); f = RoomThermalModel(PARAMS); T = 24.0; e = []
    for t in range(8, 14):
        Tf = f.step(T, out[t], 0, OCC[t], 1.0); T = m.step(T, out[t], 1, OCC[t], 1.0)
        e.append((Tf-T)*PARAMS["thermal_mass"])
    return float(np.mean(e))
def real(s, out):
    m = RoomThermalModel(TRUE); T = 24.0; v = 0
    for t in range(H):
        T = m.step(T, out[t], int(s[t]), OCC[t], 1.0); v += int(OCC[t] and abs(T-24.0) > 2.0)
    return cost(s), v
R = {"Day-ahead QUBO, wrong model": [], "Day-ahead QUBO, learned model": []}
for i in range(20):
    r = np.random.RandomState(i)
    out = [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
    p = learn(out)
    for k, pw in zip(R, [3.0, p]):
        q.PARAMS["ac_cooling_power"] = pw
        R[k].append(real(q.qubo_plan(out, 1.0), out))
    q.PARAMS["ac_cooling_power"] = 3.0
print("Real AC 1.8, planner believes 3.0, no re-planning (20 days)")
for k, v in R.items():
    v = np.array(v); print(f" {k:30s} cost Rs {v[:,0].mean():5.1f}/day  violations {v[:,1].mean():.2f} h/day")
