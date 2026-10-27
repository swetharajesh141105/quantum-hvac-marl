import numpy as np, dimod
from dwave.samplers import SimulatedAnnealingSampler
from tariff_qubo_demo import *
pn = PRICE/PRICE.max(); BLK = {14,15,16}; NOTICE = 10; LAM = 1.0
TRUE = dict(PARAMS, ac_cooling_power=1.8)
def replan(k, T0, out, P, blk, target=24.5, c_sw=0.05):
    a = P["ac_cooling_power"]/P["thermal_mass"]
    d = 1 - 1/(P["insulation_resistance"]*P["thermal_mass"])
    m = RoomThermalModel(P); T = T0; Tf = {}
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
def run(mode, out):
    room = RoomThermalModel(TRUE); T = 24.0; B = dict(PARAMS); s = []; v = 0
    for k in range(H):
        blk = BLK if k >= NOTICE else set()
        if mode == "thermo": x = int(OCC[k] and T > 25.0)
        else: x = replan(k, T, out, B, blk) if k in WINDOW else 0
        if k in BLK: x = 0
        Tfree = RoomThermalModel(B).step(T, out[k], 0, OCC[k], 1.0)
        T = room.step(T, out[k], x, OCC[k], 1.0)
        if mode == "full" and x:
            est = (Tfree - T) * B["thermal_mass"]
            B["ac_cooling_power"] = float(np.clip(0.5*B["ac_cooling_power"] + 0.5*est, 0.3, 6))
        s.append(x); v += int(OCC[k] and abs(T-24.0) > 2.0)
    return cost(s), v, B["ac_cooling_power"]
R = {"thermo": [], "wrong": [], "full": []}
for i in range(20):
    r = np.random.RandomState(i)
    out = [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
    for m in R: R[m].append(run("thermo" if m == "thermo" else ("wrong" if m == "wrong" else "full"), out))
N = {"thermo": "Thermostat", "wrong": "Grid-aware QUBO, wrong model", "full": "Full: learn + grid-aware QUBO"}
print("Real AC 1.8, believed 3.0, grid ban 14-17h announced at 10h (20 days)")
for m, v in R.items():
    v = np.array(v); print(f" {N[m]:30s} cost Rs {v[:,0].mean():5.1f}/day  violations {v[:,1].mean():.2f} h/day  power {v[:,2].mean():.2f}")
