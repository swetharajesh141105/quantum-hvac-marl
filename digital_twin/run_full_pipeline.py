"""
run_full_pipeline.py — Phase 8: full pipeline validated end-to-end in simulation.

Chains everything built so far into one continuous run:
Digital Twin -> LSTM occupancy prediction (pre-cooling) -> Federated MARL
agents -> QUBO 24h schedule optimization -> XAI plain-English explanations.
"""
import os
import numpy as np
import torch

from thermal_model import RoomThermalModel
from data_gen import generate_day_profile
from marl_agents import robust_load, joint_decision_verbose, ROOM_PARAMS
from lstm_occupancy import OccupancyLSTM, predict_occupancy_30min, train_and_save, SEQ_LEN
from qubo_scheduler import build_qubo, solve_schedule
from xai_explainer import explain_decision, explain_schedule

HOURS = 24
LSTM_PATH = "occupancy_lstm.pt"


def get_occupancy_model():
    """Load the trained LSTM occupancy predictor, training it first if the
    checkpoint doesn't exist yet."""
    if not os.path.exists(LSTM_PATH):
        print("No occupancy_lstm.pt found — training LSTM occupancy predictor first...")
        return train_and_save()
    model = OccupancyLSTM()
    model.load_state_dict(torch.load(LSTM_PATH, weights_only=True))
    model.eval()
    return model


def run_pipeline():
    print("=== STEP 1: Digital Twin — generating a simulated day ===")
    profile = generate_day_profile(seed=42)
    outdoor_temps = profile["outdoor_temp"]
    occupancy = profile["occupancy"]

    print("=== STEP 2: Loading (or training) LSTM occupancy predictor ===")
    occ_model = get_occupancy_model()

    print("=== STEP 3: Loading federated MARL agents ===")
    agents = {obj: robust_load(obj) for obj in ["energy", "comfort", "carbon"]}

    print("\n=== STEP 4: Simulating the day, hour by hour (MARL + LSTM pre-cooling) ===")
    model = RoomThermalModel(ROOM_PARAMS)
    temp = 30.0
    actions, explanations = [], []

    for t in range(HOURS):
        history = [occupancy[max(0, i)] for i in range(t - SEQ_LEN, t)]
        will_occupy, prob = predict_occupancy_30min(occ_model, history)
        predicted_occupancy = int(will_occupy)

        action, reasoning = joint_decision_verbose(
            temp=temp,
            outdoor_temp=outdoor_temps[t],
            occupancy=predicted_occupancy,   # LSTM prediction drives the decision
            t=t,
            agents=agents,
        )
        actions.append(action)
        explanations.append(explain_decision(action, reasoning))

        # Advance the digital twin using the REAL occupancy, even though the
        # decision above used the PREDICTED occupancy — this is pre-cooling.
        temp = model.step(temp, outdoor_temps[t], action, occupancy[t], dt=1.0)

        tag = " (pre-cooled ahead of arrival)" if predicted_occupancy and not occupancy[t] else ""
        print(f"Hour {t:02d}: {explanations[-1]}{tag}")

    print("\n=== STEP 5: QUBO — optimizing the 24h schedule ===")
    preferred = np.array(actions)
    bqm = build_qubo(preferred)
    schedule = solve_schedule(bqm)

    print("\n=== STEP 6: XAI — explaining the final QUBO-optimized schedule ===")
    print(explain_schedule(schedule, preferred))

    marl_savings = 100 * (1 - sum(actions) / HOURS)
    qubo_savings = 100 * (1 - sum(schedule) / HOURS)
    print(f"\n=== SUMMARY ===")
    print(f"MARL+LSTM-only savings vs always-on: {marl_savings:.1f}%")
    print(f"QUBO-optimized savings vs always-on: {qubo_savings:.1f}%")


if __name__ == "__main__":
    run_pipeline()