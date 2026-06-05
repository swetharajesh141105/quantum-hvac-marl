"""
Federated Learning — Server + Client
======================================
Implements privacy-preserving collaborative learning across
multiple "buildings" (simulated as separate FL clients).

CONCEPT (for interviews):
--------------------------
Normal ML: All buildings → send RAW DATA to cloud → train one model
Problem: Buildings won't share occupancy data (privacy, legal issues)

Federated Learning (FL):
  1. Server sends global model to all buildings
  2. Each building trains on LOCAL data (never leaves building)
  3. Buildings send only MODEL WEIGHTS (not data) to server
  4. Server aggregates weights using FedAvg algorithm
  5. Updated global model sent back to all buildings
  6. Repeat N rounds

FedAvg formula:
  w_global = Σᵢ (nᵢ/N) * wᵢ
  where nᵢ = number of samples from client i, N = total samples
  This is a weighted average of all clients' weights.

WHY IT WORKS:
Neural network weights encode the "knowledge" learned from data
without encoding the data itself. The average of weights trained
on different datasets often generalizes better than any individual
— this is a form of ensemble learning.

PRIVACY GUARANTEE:
Raw sensor readings (who was in the room, when) NEVER leave the
building. Only model weight tensors are shared.
In production, you'd add Differential Privacy (noise to weights)
for even stronger guarantees.
"""

import numpy as np
import json
import os
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ModelWeights:
    """Serializable container for neural network weights."""
    W1: list
    b1: list
    W2: list
    b2: list
    W3: list
    b3: list
    n_samples: int = 100  # number of local training samples

    def to_arrays(self):
        return {k: np.array(v) for k, v in asdict(self).items() if k != 'n_samples'}

    @classmethod
    def from_arrays(cls, arrays: dict, n_samples: int = 100):
        return cls(**{k: v.tolist() for k, v in arrays.items()}, n_samples=n_samples)


class FederatedServer:
    """
    Central aggregation server.
    
    Responsibilities:
    - Distribute global model to all clients
    - Collect local weight updates
    - Aggregate using FedAvg
    - Broadcast improved global model
    """

    def __init__(self, model_shape: Dict):
        """
        model_shape: {"input": 10, "hidden": 64, "output": 6}
        """
        self.shape = model_shape
        self.round_num = 0
        self.global_weights = self._initialize_global_model()
        self.history = []

    def _initialize_global_model(self) -> Dict:
        """Xavier-initialized global model."""
        inp, hid, out = self.shape["input"], self.shape["hidden"], self.shape["output"]
        return {
            "W1": np.random.randn(inp, hid) * np.sqrt(2 / inp),
            "b1": np.zeros(hid),
            "W2": np.random.randn(hid, hid) * np.sqrt(2 / hid),
            "b2": np.zeros(hid),
            "W3": np.random.randn(hid, out) * 0.01,
            "b3": np.zeros(out),
        }

    def get_global_model(self) -> Dict:
        """Return current global model (sent to all clients)."""
        return {k: v.copy() for k, v in self.global_weights.items()}

    def aggregate(self, client_updates: List[ModelWeights]) -> Dict:
        """
        FedAvg aggregation.
        
        Weighted average where each client's contribution is
        proportional to its local dataset size.
        
        This is federated averaging as published by McMahan et al. 2017
        "Communication-Efficient Learning of Deep Networks from
        Decentralized Data" — one of the most cited ML papers.
        """
        if not client_updates:
            return self.global_weights

        total_samples = sum(c.n_samples for c in client_updates)

        # Initialize aggregated weights as zeros
        aggregated = {k: np.zeros_like(v) for k, v in self.global_weights.items()}

        for client in client_updates:
            weight = client.n_samples / total_samples
            arrays = client.to_arrays()
            for key in aggregated:
                aggregated[key] += weight * arrays[key]

        self.global_weights = aggregated
        self.round_num += 1

        # Track history
        self.history.append({
            "round": self.round_num,
            "n_clients": len(client_updates),
            "total_samples": total_samples,
        })

        print(f"  [Server] Round {self.round_num}: aggregated {len(client_updates)} clients "
              f"({total_samples} total samples)")

        return self.global_weights


class FederatedClient:
    """
    Local building client.
    
    Each client:
    1. Receives global model from server
    2. Generates local sensor data (simulated)
    3. Trains locally for N epochs
    4. Returns weight updates (NOT the data)
    """

    def __init__(self, client_id: int, building_profile: str = "office"):
        """
        building_profile: "office", "hospital", "mall"
        Each has different occupancy patterns → different local data
        """
        self.client_id = client_id
        self.building_profile = building_profile
        self.local_data = []
        print(f"  [Client {client_id}] Initialized — Building type: {building_profile}")

    def generate_local_data(self, n_samples: int = 200) -> List[Tuple]:
        """
        Simulate local sensor data collection.
        
        In real deployment, this would be actual sensor readings
        from THIS building's DHT22, PIR, CO2 sensors.
        
        Different building profiles → different patterns → richer global model
        """
        from digital_twin.thermal_model import DigitalTwin, RoomConfig

        profiles = {
            "office":   RoomConfig(area_sqm=20, max_occupancy=10),
            "hospital": RoomConfig(area_sqm=30, max_occupancy=5),   # always occupied
            "mall":     RoomConfig(area_sqm=100, max_occupancy=50),  # high variance
        }

        config = profiles.get(self.building_profile, RoomConfig())
        twin = DigitalTwin(config=config, seed=self.client_id * 42)

        data = []
        obs = twin.reset()
        for _ in range(n_samples):
            # Use heuristic "expert" actions as training labels
            action = self._heuristic_action(twin)
            new_obs, reward, done, info = twin.step(action)
            data.append((obs.copy(), action, reward))
            obs = new_obs
            if done:
                obs = twin.reset()

        self.local_data = data
        return data

    def _heuristic_action(self, twin) -> int:
        """
        Rule-based expert policy for generating training labels.
        
        This is supervised pre-training data. The RL fine-tunes from here.
        Rules: simple human intuition about HVAC.
        """
        if twin.occupants == 0:
            return 0  # OFF when empty

        if twin.temp > 26:
            return 2  # Cool aggressively to 20°C

        if twin.temp > 24:
            return 3  # Cool to 22°C

        if twin.temp > 22:
            return 4  # Mild cooling to 24°C

        return 5  # Very mild / maintain 26°C

    def local_train(
        self,
        global_weights: Dict,
        local_epochs: int = 5,
        learning_rate: float = 0.001
    ) -> ModelWeights:
        """
        Train on local data starting from global model weights.
        
        This is standard supervised learning on local sensor data.
        We train for a few epochs then return the updated weights.
        
        In production with Flower framework:
            def fit(self, parameters, config):
                self.model.set_parameters(parameters)
                self.model.fit(self.local_dataset)
                return self.model.get_parameters(), len(self.local_dataset), {}
        """
        # Copy global weights
        local_W = {k: v.copy() for k, v in global_weights.items()}

        if not self.local_data:
            self.generate_local_data()

        for epoch in range(local_epochs):
            total_loss = 0
            np.random.shuffle(self.local_data)

            for obs, target_action, reward in self.local_data:
                # Forward pass
                h1 = np.tanh(obs @ local_W["W1"] + local_W["b1"])
                h2 = np.tanh(h1 @ local_W["W2"] + local_W["b2"])
                logits = h2 @ local_W["W3"] + local_W["b3"]
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / exp_l.sum()

                # Cross-entropy loss weighted by reward
                loss = -np.log(probs[target_action] + 1e-8) * max(0, reward)
                total_loss += loss

                # Backward pass (simplified gradient)
                d_logits = probs.copy()
                d_logits[target_action] -= 1.0
                d_logits *= max(0, reward) * learning_rate

                local_W["W3"] -= np.outer(h2, d_logits)
                local_W["b3"] -= d_logits

            if epoch == local_epochs - 1:
                avg_loss = total_loss / len(self.local_data)
                # print(f"    Client {self.client_id} | Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        return ModelWeights.from_arrays(local_W, n_samples=len(self.local_data))


# ------------------------------------------------------------------ #
#  Full Federated Training Simulation                                  #
# ------------------------------------------------------------------ #

def run_federated_training(
    n_rounds: int = 10,
    n_clients: int = 3,
    local_epochs: int = 5
):
    """
    Simulate a complete federated learning session.
    
    n_rounds: how many communication rounds (global aggregations)
    n_clients: number of "buildings" participating
    local_epochs: how long each client trains before sending weights
    """
    print("=" * 60)
    print("FEDERATED LEARNING SIMULATION")
    print(f"Rounds: {n_rounds} | Clients: {n_clients} | Local epochs: {local_epochs}")
    print("=" * 60)

    # Initialize server
    model_shape = {"input": 10, "hidden": 64, "output": 6}
    server = FederatedServer(model_shape)

    # Initialize clients with different building types
    building_types = ["office", "hospital", "mall"]
    clients = [
        FederatedClient(i, building_types[i % len(building_types)])
        for i in range(n_clients)
    ]

    # Pre-generate local data for each client
    print("\nGenerating local datasets...")
    for client in clients:
        client.generate_local_data(n_samples=150)
        print(f"  Client {client.client_id} ({client.building_profile}): "
              f"{len(client.local_data)} samples")

    # Federated training rounds
    print(f"\nStarting {n_rounds} federation rounds...")
    for round_num in range(1, n_rounds + 1):
        print(f"\n[Round {round_num}/{n_rounds}]")

        # 1. Server broadcasts global model
        global_model = server.get_global_model()

        # 2. Each client trains locally
        client_updates = []
        for client in clients:
            weights = client.local_train(global_model, local_epochs=local_epochs)
            client_updates.append(weights)
            print(f"  Client {client.client_id} ✓ local training done")

        # 3. Server aggregates
        server.aggregate(client_updates)

    # Save final global model
    final_model = server.get_global_model()
    save_path = "federated_global_model.npz"
    np.savez(save_path, **final_model)
    print(f"\n✅ Federated training complete! Global model saved to {save_path}")
    print(f"   {n_rounds} rounds × {n_clients} clients = {n_rounds * n_clients} local trainings")
    print(f"   Each client's raw data NEVER left its building 🔒")

    return final_model


if __name__ == "__main__":
    final_model = run_federated_training(n_rounds=5, n_clients=3)
    print("\nFL training complete. Global model ready for deployment on edge devices.")
