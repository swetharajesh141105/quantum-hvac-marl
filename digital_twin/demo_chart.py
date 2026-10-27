import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tariff_qubo_demo import *
rng = np.random.RandomState(0)
out = [24+8*np.sin((h-6)*np.pi/12)+rng.normal(0,1) for h in range(H)]
b = thermostat(out); q = best_plan(out, simulate(b, out))
def temps(s):
    m = RoomThermalModel(PARAMS); T = 24.0; r = []
    for t in range(H): T = m.step(T, out[t], int(s[t]), OCC[t], 1.0); r.append(T)
    return r
tb, tq = temps(b), temps(q); h = np.arange(H)
f, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
a1.bar(h-.2, b, .4, color="#95a5a6", label="Thermostat ON")
a1.bar(h+.2, q, .4, color="#1d8a99", label="QUBO plan ON")
a1.set_yticks([]); a1.legend(loc="upper left", fontsize=8)
p = a1.twinx(); p.step(h, PRICE, "r-", where="mid"); p.set_ylabel("Tariff (Rs/kWh)", color="r")
a1.set_title(f"One example day: thermostat Rs {cost(b):.0f} vs QUBO Rs {cost(q):.0f}")
a2.plot(h, tb, "o-", color="#95a5a6", label="Thermostat temp")
a2.plot(h, tq, "o-", color="#1d8a99", label="QUBO temp")
a2.axhspan(22, 26, color="green", alpha=.12, label="Comfort band 22-26C")
for t in range(H):
    if OCC[t]: a2.axvspan(t-.5, t+.5, color="orange", alpha=.08)
a2.set_xlabel("Hour (orange = room occupied)"); a2.set_ylabel("Room temp (C)")
a2.legend(loc="upper left", fontsize=8); a2.set_xticks(range(0, 24, 2))
plt.tight_layout(); plt.savefig("demo_chart.png", dpi=200)
print("Saved demo_chart.png")
