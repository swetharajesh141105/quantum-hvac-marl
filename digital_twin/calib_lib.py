import numpy as np, dimod
from dwave.samplers import SimulatedAnnealingSampler
from tariff_qubo_demo import *
pn = PRICE/PRICE.max(); LAM = 1.0
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
def ci(x): return 1.96*np.std(x, ddof=1)/np.sqrt(len(x))
