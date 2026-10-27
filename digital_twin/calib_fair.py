import numpy as np
from calib_lib import *
def run(mode, out, tp, ban):
    probe = mode in ("wrong_probe", "full", "oracle_probe")
    room = RoomThermalModel(dict(PARAMS, ac_cooling_power=tp)); T = 24.0
    B = dict(PARAMS, ac_cooling_power=tp) if mode == "oracle_probe" else dict(PARAMS)
    v = 0; c = 0.0
    for k in range(H):
        blk = ban if k >= 10 else set()
        if probe and k == 8: x = 1
        elif k in WINDOW: x = replan(k, T, out, B, blk)
        else: x = 0
        if k in ban: x = 0
        Tfree = RoomThermalModel(B).step(T, out[k], 0, OCC[k], 1.0)
        T = room.step(T, out[k], x, OCC[k], 1.0); c += PRICE[k]*KW*x
        if mode == "full" and x:
            est = (Tfree - T) * B["thermal_mass"]
            B["ac_cooling_power"] = float(np.clip(0.5*B["ac_cooling_power"] + 0.5*est, 0.3, 6))
        v += int(OCC[k] and abs(T-24.0) > 2.0)
    return c, v
N = 30
print(f"Fair test, {N} days. All 3 probe rows get the same hour-8 probe. mean +/- 95% CI")
for tp, ban in [(2.4,{15,16}),(1.8,{15,16}),(2.4,{14,15,16})]:
    R = {m: [] for m in ("wrong_probe","full","oracle_probe")}
    for i in range(N):
        r = np.random.RandomState(i)
        out = [24+8*np.sin((h-6)*np.pi/12)+r.normal(0,1) for h in range(H)]
        for m in R: R[m].append(run(m, out, tp, ban))
    print(f"\ntrue AC {tp}, ban {sorted(ban)}")
    for m, v in R.items():
        v = np.array(v); print(f"  {m:12s} violations {v[:,1].mean():.2f} +/- {ci(v[:,1]):.2f}  cost Rs {v[:,0].mean():.1f} +/- {ci(v[:,0]):.1f}")
    d = np.array(R["wrong_probe"])[:,1] - np.array(R["full"])[:,1]
    print(f"  PURE learning effect: violations cut by {d.mean():.2f} +/- {ci(d):.2f} h/day")
