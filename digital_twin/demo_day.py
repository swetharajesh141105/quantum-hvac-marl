"""Panel demo: ONE day, hour by hour. Thermostat vs QUBO plan, with price and room temperature."""
import numpy as np
from tariff_qubo_demo import *
rng = np.random.RandomState(0)
out = [24+8*np.sin((h-6)*np.pi/12)+rng.normal(0,1) for h in range(H)]
b = thermostat(out); q = best_plan(out, simulate(b, out))
def temps(s):
    m = RoomThermalModel(PARAMS); T = 24.0; r = []
    for t in range(H): T = m.step(T, out[t], int(s[t]), OCC[t], 1.0); r.append(T)
    return r
tb, tq = temps(b), temps(q)
print("Hour  Occupied  Price   Thermostat(temp)    QUBO plan(temp)")
for t in range(7, 19):
    print(f"{t:02d}:00   {'Yes' if OCC[t] else 'No ':3s}     Rs{PRICE[t]:>2.0f}    {'ON ' if b[t] else 'OFF'} ({tb[t]:.1f}C)        {'ON ' if q[t] else 'OFF'} ({tq[t]:.1f}C)")
print(f"\nDay cost: thermostat Rs {cost(b):.1f}  |  QUBO plan Rs {cost(q):.1f}")
print(f"Comfort violations: thermostat {simulate(b,out)} h  |  QUBO plan {simulate(q,out)} h")
