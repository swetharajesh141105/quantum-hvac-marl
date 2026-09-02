"""
QUBO Global Scheduler.
Step 1: get the federated agents' preferred ON/OFF action for each hour of the day.
Step 2: formulate a QUBO that finds the best 24h schedule — minimizing energy use
         and switching, while staying close to what the agents learned.
Step 3: solve with quantum-inspired simulated annealing (neal).
"""
import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from stable_baselines3 import PPO
from hvac_env import HVACEnv
from marl_agents import joint_decision, robust_load
from data_gen import generate_day_profile

HOURS = 24

def load_federated_agents():
    agents = {}
    for obj in ["energy", "comfort", "carbon"]:
        agents[obj] = robust_load(obj)
    return agents

def get_preferred_actions(agents):
    profile = generate_day_profile(seed=42)
    outdoor_temps_full = profile["outdoor_temp"]
    occupancy_full = profile["occupancy"]
    step = len(outdoor_temps_full) // HOURS
    outdoor_temps = [outdoor_temps_full[i * step] for i in range(HOURS)]
    occupancy = [occupancy_full[i * step] for i in range(HOURS)]
    preferred = []
    temp = 30.0
    for t in range(HOURS):
        action = joint_decision(temp, outdoor_temps[t], occupancy[t], t, agents)
        preferred.append(action)
        temp += (outdoor_temps[t] - temp) * 0.1 - action * 1.5
    return np.array(preferred)

def solve_schedule(bqm, num_reads=100):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first.sample
    schedule = [best[t] for t in range(HOURS)]
    return schedule

if __name__ == "__main__":
    print("Loading federated agents...")
    agents = load_federated_agents()

    print("Getting agents' preferred 24h actions...")
    preferred = get_preferred_actions(agents)
    print("Preferred (agent-only):", preferred.tolist())

    print("Building QUBO...")
    bqm = build_qubo(preferred)

    print("Solving with quantum-inspired simulated annealing...")
    schedule = solve_schedule(bqm)
    print("Final QUBO-optimized 24h schedule:", schedule)

    energy_savings = 100 * (1 - sum(schedule) / HOURS)
    print(f"AC ON hours: {sum(schedule)}/24 | Energy savings vs always-on: {energy_savings:.1f}%")
import dimod

def build_qubo(preferred_actions, c_energy=0.8, c_deviation=0.3, c_switch=0.4):
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    HOURS = len(preferred_actions)
    for t in range(HOURS):
        pref = preferred_actions[t]
        bqm.add_linear(t, c_energy)
        bqm.add_linear(t, c_deviation * (1 - 2 * pref))
    for t in range(HOURS - 1):
        bqm.add_linear(t, c_switch)
        bqm.add_linear(t + 1, c_switch)
        bqm.add_quadratic(t, t + 1, -2 * c_switch)
    return bqm
