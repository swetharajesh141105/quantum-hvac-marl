import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tariff_qubo_demo import *
def day(s):
    r = np.random.RandomState(s)
    return [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
days = [day(s) for s in range(30)]
base = [thermostat(o) for o in days]
bc = np.mean([cost(b) for b in base]); bv = np.mean([simulate(b,o) for b,o in zip(base,days)])
P = []
for lam in [0,0.3,1,2,4,6]:
    pl = [qubo_plan(o,lam) for o in days]
    P.append((np.mean([simulate(p,o) for p,o in zip(pl,days)]), np.mean([cost(p) for p in pl]), lam))
S = [0,1,2,3]; sav = []; vio = []
for sg in S:
    c = []; v = []
    for i,o in enumerate(days):
        fc = list(np.array(o) + np.random.RandomState(100+i).normal(0,sg,H))
        q = best_plan(fc, simulate(base[i],o))
        c.append(cost(q)); v.append(simulate(q,o))
    sav.append(100*(1-np.mean(c)/bc)); vio.append(np.mean(v))
    print(f"Forecast error {sg}C: saving {sav[-1]:.1f}%  violations {vio[-1]:.2f} h/day")
f, (a, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
a.plot([p[0] for p in P], [p[1] for p in P], "o-", color="#1d8a99", label="QUBO (weight sweep)")
a.plot(bv, bc, "r*", ms=16, label="Thermostat")
a.set_xlabel("Comfort violations (h/day)"); a.set_ylabel("Cost (Rs/day)")
a.set_title("QUBO gives a cost-comfort dial"); a.legend()
b2.bar([str(s) for s in S], sav, color="#1d8a99")
for i,(s,v) in enumerate(zip(sav,vio)): b2.text(i, s, f"{v:.2f}h viol", ha="center", va="bottom", fontsize=8)
b2.set_xlabel("Weather-forecast error (C, std)"); b2.set_ylabel("Cost saving vs thermostat (%)")
b2.set_title("Savings with a wrong forecast")
plt.tight_layout(); plt.savefig("wow_chart.png", dpi=200); print("Saved wow_chart.png")
