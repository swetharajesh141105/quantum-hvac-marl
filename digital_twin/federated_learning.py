"""
Federated Learning wrapper for MARL agents.
Simulates 3 buildings training locally, then averages (FedAvg) their
policy network WEIGHTS only — raw sensor data never leaves each building.
"""
from stable_baselines3 import PPO
from hvac_env import HVACEnv
from marl_agents import SingleObjectiveWrapper
import copy

# 3 simulated "buildings" — different room physics, same agent architecture
BUILDING_PARAMS = [
    {"thermal_mass": 5.0, "insulation_resistance": 2.0, "ac_cooling_power": 3.0, "occupancy_heat_gain": 0.5},
    {"thermal_mass": 6.0, "insulation_resistance": 1.5, "ac_cooling_power": 3.5, "occupancy_heat_gain": 0.6},
    {"thermal_mass": 4.5, "insulation_resistance": 2.5, "ac_cooling_power": 2.8, "occupancy_heat_gain": 0.4},
]

NUM_ROUNDS = 5
LOCAL_TIMESTEPS = 5000  # more training per round

def create_local_model(objective, room_params, init_state_dict=None):
    env = SingleObjectiveWrapper(HVACEnv(room_params), objective)
    model = PPO("MlpPolicy", env, verbose=0, ent_coef=0.01)
    if init_state_dict is not None:
        model.policy.load_state_dict(init_state_dict)
    return model

def average_state_dicts(state_dicts):
    """FedAvg: mean of weights across all buildings."""
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        for sd in state_dicts[1:]:
            avg[key] += sd[key]
        avg[key] = avg[key] / len(state_dicts)
    return avg

def federated_train(objective):
    global_state_dict = None
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== {objective.upper()} | Round {rnd}/{NUM_ROUNDS} ===")
        local_state_dicts = []
        for i, params in enumerate(BUILDING_PARAMS):
            print(f"  Training Building {i+1} locally...")
            model = create_local_model(objective, params, global_state_dict)
            model.learn(total_timesteps=LOCAL_TIMESTEPS)
            local_state_dicts.append(model.policy.state_dict())
        global_state_dict = average_state_dicts(local_state_dicts)  # server aggregates weights only

    final_model = create_local_model(objective, BUILDING_PARAMS[0], global_state_dict)
    final_model.save(f"{objective}_agent_federated")
    print(f"Saved {objective}_agent_federated.zip")
    return final_model

if __name__ == "__main__":
    for objective in ["energy", "comfort", "carbon"]:
        federated_train(objective)
    print("\nFederated training complete for all 3 agents.")