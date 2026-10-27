import numpy as np
from marl_agents import robust_load, ROOM_PARAMS
from tariff_qubo_demo import *
agents = {k: robust_load(k) for k in ("energy", "carbon", "comfort")}
def marl(out, safe):
    m = RoomThermalModel(ROOM_PARAMS); T = 24.0; s = []
    for t in range(H):
        gap = next((i for i in range(t+1, H) if OCC[i]), None)
        pre = int(not OCC[t] and gap == t+1 and abs(m.step(T, out[t], 0, 0, 1.0)-24.0) > 1.5)
        sig = int(OCC[t] or pre)
        obs = np.array([T, out[t], sig, t, 400.0], dtype=np.float32)
        v = sum(int(agents[k].predict(obs, deterministic=True)[0]) for k in agents)
        if sig:
            a = 1 if T > 25.5 else (0 if T < 22.5 else int(v >= 2))
        else: a = 0
        if safe and OCC[t] and m.step(T, out[t], 0, OCC[t], 1.0) > 25.5: a = 1
        T = m.step(T, out[t], a, OCC[t], 1.0); s.append(a)
    return np.array(s)
def ci(x): return 1.96*np.std(x, ddof=1)/np.sqrt(len(x))
R = {"Thermostat": [], "MARL (original)": [], "MARL + predictive safety": [], "Tariff-aware QUBO": []}
for i in range(30):
    r = np.random.RandomState(i)
    out = [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
    b = thermostat(out)
    for k, s in zip(R, [b, marl(out, False), marl(out, True), best_plan(out, simulate(b, out))]):
        R[k].append((cost(s), simulate(s, out)))
print("30 noisy days, assumed tariff. mean +/- 95% CI")
for k, v in R.items():
    v = np.array(v); print(f" {k:26s} cost Rs {v[:,0].mean():5.1f} +/- {ci(v[:,0]):.1f}   violations {v[:,1].mean():.2f} +/- {ci(v[:,1]):.2f} h/day")
