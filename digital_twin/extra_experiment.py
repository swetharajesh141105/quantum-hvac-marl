import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tariff_qubo_demo as q

W = json.load(open("vellore_weather.json"))
days = [W[i*24:(i+1)*24] for i in range(len(W)//24)]
FC = [list(np.array(d) + np.random.RandomState(i).normal(0, 2.0, 24)) for i, d in enumerate(days)]
TN = np.array([8.645]*5+[9.10]+[11.375]*4+[9.10]*8+[11.375]*4+[8.645]*2)
HYP = q.PRICE.copy(); FLAT = np.full(24, TN.mean())
ci = lambda x: 1.96*np.std(x, ddof=1)/np.sqrt(len(x))
idx = np.arange(2**13)
S = np.zeros((len(idx), 24)); S[:, 5:18] = (idx[:, None] >> np.arange(13)) & 1
CACHE = {}
def vb(key, ds):  # comfort violations of all 8192 plans, per day (cached)
    if key not in CACHE: CACHE[key] = [np.array([q.simulate(s, d) for s in S]) for d in ds]
    return CACHE[key]

def run(price, cap, sg=0.0, sel=range(30)):
    q.PARAMS["ac_cooling_power"] = cap; q.KW = cap/2; q.PRICE = price
    fc = days if sg == 0 else FC
    VBf = vb(("f", cap, sg), fc)
    cb = S @ (price*q.KW); lams = np.geomspace(100, 0.1, 25)*(cap/3)**2
    R = {m: [] for m in ("Thermostat", "QUBO (SA)", "Brute-force optimum")}
    for i in sel:
        d, f = days[i], fc[i]
        b = q.thermostat(d); c0 = q.cost(b)
        v0 = q.simulate(q.thermostat(f), f)            # comfort budget the planner knows
        ok = np.where(VBf[i] <= v0)[0]; best = S[ok[np.argmin(cb[ok])]]
        for lam in lams:
            u = q.qubo_plan(f, lam)
            if q.simulate(u, f) <= v0: break
        for m, s in (("Thermostat", b), ("QUBO (SA)", u), ("Brute-force optimum", best)):
            c = q.cost(s)   # cost and comfort are always measured on the REAL weather
            R[m].append((c, q.simulate(s, d), 100*(1-c/c0) if c0 > 0 else 0.0, float(np.sum(s))*q.KW))
    return R

OUT = ["setting,method,cost_Rs_day,uncomfortable_h_day,kWh_day,saving_pct,ci95"]
def show(tag, R):
    for m, r in R.items():
        c, v, sv, k = (np.array(x) for x in zip(*r))
        print(f"{tag:30s} {m:20s} cost {c.mean():6.1f} uncomf {v.mean():.2f} kWh {k.mean():5.1f} saving {sv.mean():6.1f}% +/- {ci(sv):.1f}", flush=True)
        OUT.append(f"{tag},{m},{c.mean():.1f},{v.mean():.2f},{k.mean():.1f},{sv.mean():.1f},{ci(sv):.1f}")

for tn, p in (("TNERC", TN), ("Hypothetical", HYP), ("Flat control", FLAT)):
    show(f"{tn} AC9 perfect-fc", run(p, 9))
for tn, p in (("TNERC", TN), ("Hypothetical", HYP)):
    show(f"{tn} AC9 fc-error-2C", run(p, 9, 2.0))
    show(f"{tn} AC12 perfect-fc", run(p, 12))

# break-even: scale the peak/night gap, keep the mean price fixed (10 days for speed)
m_ = TN.mean(); X, Yq, Yb = [], [], []
for k in (0, 0.5, 1, 2, 3, 4, 6):
    p = m_ + k*(TN - m_); R = run(p, 9, 0.0, range(0, 30, 3)); X.append(p.max()/p.min())
    Yq.append([x[2] for x in R["QUBO (SA)"]]); Yb.append([x[2] for x in R["Brute-force optimum"]])
    print(f"k={k} ratio={X[-1]:.2f} QUBO {np.mean(Yq[-1]):.1f}%  optimum {np.mean(Yb[-1]):.1f}%", flush=True)
    OUT.append(f"breakeven k={k} ratio={X[-1]:.2f},QUBO,,,,{np.mean(Yq[-1]):.1f},{ci(np.array(Yq[-1])):.1f}")
for Y, lab in ((Yq, "QUBO (SA)"), (Yb, "Brute-force optimum")):
    plt.errorbar(X, [np.mean(y) for y in Y], yerr=[ci(np.array(y)) for y in Y], marker="o", capsize=3, label=lab)
plt.axvline(TN.max()/TN.min(), color="r", ls="--", label="TNERC tariff")
plt.axhline(0, color="k", lw=0.8); plt.xlabel("Peak / night price ratio (mean price fixed)")
plt.ylabel("Cost saving vs thermostat (%)"); plt.legend(); plt.grid(alpha=0.3)
plt.savefig("breakeven_final.png", dpi=200, bbox_inches="tight")
open("extra_results.csv", "w").write("\n".join(OUT)); print("saved extra_results.csv and breakeven_final.png")
