import numpy as np
from stable_baselines3 import PPO
from thermal_model import RoomThermalModel
from marl_agents import ROOM_PARAMS

OCC=[0]*9+[1]*4+[0]+[1]*4+[0]*6
def load(suffix): return {o:PPO.load(f"{o}_agent{suffix}") for o in ["energy","carbon","comfort"]}
def obs(T,out,occ,t): return np.array([T,out,occ,t,400.0],dtype=np.float32)

for name,suffix in [("LOCAL",""),("FEDERATED","_federated")]:
    ag=load(suffix); print(f"\n=== CHECK 1: does each {name} agent ever vote ON? ===")
    for o,m in ag.items():
        on=sum(int(m.predict(obs(T,30,occ,12),deterministic=True)[0]) for T in range(20,31) for occ in (0,1))
        print(f" {o:8s} ON votes: {on}/22  ->", "OK" if on>0 else "COLLAPSED (always OFF)")

print("\n=== CHECK 2: 30-day test, thermostat vs pure 3-agent majority vote (no hidden rule) ===")
ag=load("")
def run(policy,seed):
    rng=np.random.RandomState(seed); m=RoomThermalModel(ROOM_PARAMS); T=24.0; on=0; viol=0
    for t in range(24):
        out=24+8*np.sin((t-6)*np.pi/12)+rng.normal(0,1); a=policy(T,out,OCC[t],t)
        T=m.step(T,out,a,OCC[t],1.0); on+=a; viol+= int(OCC[t] and abs(T-24)>2)
    return 100*(1-on/24),viol
thermo=lambda T,o,occ,t: int(occ and T>25.0)
vote=lambda T,o,occ,t: int(sum(int(m.predict(obs(T,o,occ,t),deterministic=True)[0]) for m in ag.values())>=2)
for n,p in [("Thermostat",thermo),("Agents only",vote)]:
    r=np.array([run(p,s) for s in range(30)]); print(f" {n:12s} saving {r[:,0].mean():.1f}%   comfort-violation hrs/day {r[:,1].mean():.2f}")
