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
from marl_agents import joint_decision
from data_gen import generate_day_profile

HOURS = 24

def load_federated_agents():
    agents = {}
    for obj in ["energy", "comfort", "carbon"]:
        agents[obj] = PPO.load(f"{obj}_agent_federated")
    return agents

def get_preferred_actions(agents):
    """Run federated agents across a simulated day to get their preferred ON/OFF per hour."""
    profile = generate_day_profile(seed=42)
    outdoor_temps = profile["outdoor_temp"]
    occupancy = profile["occupancy"]

    preferred = []
    temp = 30.0
    for t in range(HOURS):
        action = joint_decision(temp, outdoor_temps[t], occupancy[t], t, agents)
        preferred.append(action)
        temp += (outdoor_temps[t] - temp) * 0.1 - action * 1.5  # rough temp update for the loop
    return np.array(preferred)

def build_qubo(preferred_actions, c_energy=0.3, c_deviation=1.0, c_switch=0.5):
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    for t in range(HOURS):
        pref = preferred_actions[t]
        # energy cost: prefer OFF (x=0) to save energy
        bqm.add_linear(t, c_energy)
        # deviation penalty: (x_t - pref)^2 = x_t - 2*pref*x_t + pref^2 -> keep linear part
        bqm.add_linear(t, c_deviation * (1 - 2 * pref))

    for t in range(HOURS - 1):
        # switching penalty: (x_t - x_{t+1})^2 = x_t + x_{t+1} - 2*x_t*x_{t+1}
        bqm.add_linear(t, c_switch)
        bqm.add_linear(t + 1, c_switch)
        bqm.add_quadratic(t, t + 1, -2 * c_switch)

    return bqm

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