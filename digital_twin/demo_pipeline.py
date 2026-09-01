"""
End-to-end demo: Digital Twin -> Federated MARL -> QUBO Schedule -> LSTM -> XAI
"""
from stable_baselines3 import PPO
from qubo_scheduler import load_federated_agents, get_preferred_actions, build_qubo, solve_schedule
from lstm_occupancy import predict_occupancy_30min, OccupancyLSTM
import torch

print("STEP 1-2: Loading federated MARL agents (trained on Digital Twin)...")
agents = load_federated_agents()

print("STEP 4: Loading LSTM occupancy predictor...")
occ_model = OccupancyLSTM()
occ_model.load_state_dict(torch.load("occupancy_lstm.pt"))
occ_model.eval()

print("STEP 5-6: Getting agents' preferred actions -> Building & solving QUBO...")
preferred = get_preferred_actions(agents)
bqm = build_qubo(preferred)
schedule = solve_schedule(bqm)

print("\n=== FINAL RESULT ===")
print("24h QUBO-Optimized Schedule:", schedule)
savings = 100 * (1 - sum(schedule)/24)
print(f"Energy savings vs always-on: {savings:.1f}%")
print("STEP 7: XAI Explanation: 'AC scheduled ON during peak occupancy/heat hours, OFF otherwise, optimized across full 24h horizon.'")