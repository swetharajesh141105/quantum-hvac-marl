"""NEW FEATURE: Tariff-aware QUBO pre-cooling for ONE building.
A thermostat reacts hour by hour. QUBO plans the whole day: it shifts AC running
away from expensive peak hours while the room stays comfortable. Standalone file."""
import numpy as np, dimod
from dwave.samplers import SimulatedAnnealingSampler
from thermal_model import RoomThermalModel

H = 24
PARAMS = {"thermal_mass":5.0,"insulation_resistance":2.0,"ac_cooling_power":3.0,"occupancy_heat_gain":0.5}
OCC = [0]*9 + [1]*4 + [0] + [1]*4 + [0]*6
KW = 1.5
PRICE = np.array([4]*9 + [7]*3 + [10]*5 + [7]*2 + [4]*5, dtype=float)   # Rs/kWh (assumed tariff)
WINDOW = range(5, 18)

def simulate(sched, outdoor):
    m = RoomThermalModel(PARAMS); T = 24.0; viol = 0
    for t in range(H):
        T = m.step(T, outdoor[t], int(sched[t]), OCC[t], 1.0)
        viol += int(OCC[t] and abs(T - 24.0) > 2.0)
    return viol

def cost(sched): return float(np.sum(PRICE * KW * np.array(sched)))

def thermostat(outdoor):
    m = RoomThermalModel(PARAMS); T = 24.0; s = []
    for t in range(H):
        a = int(OCC[t] and T > 25.0); s.append(a)
        T = m.step(T, outdoor[t], a, OCC[t], 1.0)
    return np.array(s)

def free_response(outdoor):
    m = RoomThermalModel(PARAMS); T = 24.0; out = []
    for t in range(H):
        T = m.step(T, outdoor[t], 0, OCC[t], 1.0); out.append(T)
    return np.array(out)

def qubo_plan(outdoor, lam, target=24.5, c_sw=0.05):
    """Room model is LINEAR in AC decisions, so comfort is an exact QUBO term:
    min sum_s occ_s*(T_s(x)-target)^2 + lam*sum_t price_t*x_t + switching."""
    a = PARAMS["ac_cooling_power"]/PARAMS["thermal_mass"]
    d = 1 - 1/(PARAMS["insulation_resistance"]*PARAMS["thermal_mass"])
    Tf = free_response(outdoor); pn = PRICE/PRICE.max()
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for t in range(H): bqm.add_variable(t, (lam*pn[t]) if t in WINDOW else 50.0)
    for s in range(H):
        if not OCC[s]: continue
        r = Tf[s] - target
        co = {t: a*d**(s-t) for t in WINDOW if t <= s}
        for t, ct in co.items(): bqm.add_linear(t, -2*r*ct + ct*ct)
        ts = list(co)
        for i in range(len(ts)):
            for j in range(i+1, len(ts)): bqm.add_quadratic(ts[i], ts[j], 2*co[ts[i]]*co[ts[j]])
    for t in range(H-1):
        bqm.add_linear(t, c_sw); bqm.add_linear(t+1, c_sw); bqm.add_quadratic(t, t+1, -2*c_sw)
    s = SimulatedAnnealingSampler().sample(bqm, num_reads=200, seed=1).first.sample
    return np.array([s[t] for t in range(H)])

def best_plan(outdoor, max_viol):
    for lam in [6, 4, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0]:
        s = qubo_plan(outdoor, lam)
        if simulate(s, outdoor) <= max_viol: return s
    return s

if __name__ == "__main__":
    base_c, qubo_c, base_v, qubo_v = [], [], [], []
    for seed in range(30):
        rng = np.random.RandomState(seed)
        out = [24 + 8*np.sin((h-6)*np.pi/12) + rng.normal(0, 1.0) for h in range(H)]
        b = thermostat(out); q = best_plan(out, simulate(b, out))
        base_c.append(cost(b)); qubo_c.append(cost(q)); base_v.append(simulate(b, out)); qubo_v.append(simulate(q, out))
    base_c, qubo_c = np.array(base_c), np.array(qubo_c)
    sav = 100 * (1 - qubo_c.mean() / base_c.mean())
    ci = 1.96 * np.std(100*(1-qubo_c/base_c), ddof=1) / np.sqrt(30)
    print("=== Tariff-aware QUBO pre-cooling vs reactive thermostat (30 days) ===")
    print(f" Thermostat : cost Rs {base_c.mean():6.1f}/day   comfort violations {np.mean(base_v):.2f} h/day")
    print(f" QUBO plan  : cost Rs {qubo_c.mean():6.1f}/day   comfort violations {np.mean(qubo_v):.2f} h/day")
    print(f" Electricity-cost saving from QUBO: {sav:.1f}% (+/- {ci:.1f}, 95% CI)")
    print(" Note: assumes a day-ahead weather forecast (perfect-forecast assumption).")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rng = np.random.RandomState(0); out = [24+8*np.sin((h-6)*np.pi/12)+rng.normal(0,1) for h in range(H)]
        b = thermostat(out); q = best_plan(out, simulate(b, out)); h = np.arange(H)
        f, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(h-.2, b, .4, label="Thermostat ON", color="#95a5a6"); ax.bar(h+.2, q, .4, label="QUBO plan ON", color="#1d8a99")
        ax2 = ax.twinx(); ax2.step(h, PRICE, "r-", where="mid", label="Tariff (Rs/kWh)")
        ax.set_xlabel("Hour"); ax.set_yticks([]); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Tariff-aware QUBO pre-cooling: {sav:.1f}% lower electricity cost"); plt.tight_layout(); plt.savefig("tariff_qubo_result.png", dpi=200)
        print(" Saved chart: tariff_qubo_result.png")
    except Exception as e: print("chart skipped:", e)
