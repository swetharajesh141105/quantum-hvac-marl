import json, urllib.request, numpy as np
import tariff_qubo_demo as q
# Real tariff: put 24 numbers (Rs/kWh, hour 0 to 23) from your electricity board's tariff order.
# Leave as None to keep the ASSUMED tariff.
TARIFF = None
if TARIFF: q.PRICE = np.array(TARIFF, dtype=float)
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=12.92&longitude=79.13"
       "&start_date=2025-04-01&end_date=2025-04-30&hourly=temperature_2m&timezone=Asia%2FKolkata")
try:
    T = json.load(open("vellore_weather.json"))
except Exception:
    T = json.load(urllib.request.urlopen(URL, timeout=60))["hourly"]["temperature_2m"]
    json.dump(T, open("vellore_weather.json", "w"))
days = [T[i*24:(i+1)*24] for i in range(len(T)//24)]
print(f"{len(days)} real days, mean outdoor temp {np.mean(T):.1f}C, max {max(T):.1f}C")
def ci(x): return 1.96*np.std(x, ddof=1)/np.sqrt(len(x))
for sg in (0.0, 2.0):
    C = {"thermo": [], "qubo": []}; V = {"thermo": [], "qubo": []}; S = []
    for i, real in enumerate(days):
        fc = list(np.array(real) + np.random.RandomState(i).normal(0, sg, q.H)) if sg else real
        b = q.thermostat(real); p = q.best_plan(fc, q.simulate(q.thermostat(fc), fc))
        for k, s in (("thermo", b), ("qubo", p)): C[k].append(q.cost(s)); V[k].append(q.simulate(s, real))
        S.append(100*(1-C["qubo"][-1]/C["thermo"][-1]))
        print("day", i, flush=True)
    print(f"\nREAL Vellore weather, forecast error {sg}C, {len(days)} days (mean +/- 95% CI)")
    for k in C: print(f" {k:7s} cost Rs {np.mean(C[k]):.1f} +/- {ci(C[k]):.1f}   violations {np.mean(V[k]):.2f} +/- {ci(V[k]):.2f} h/day")
    print(f" saving from QUBO: {np.mean(S):.1f}% +/- {ci(S):.1f}")
