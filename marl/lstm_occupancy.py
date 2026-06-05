"""
LSTM Occupancy Predictor
=========================
Predicts room occupancy for next 30 minutes using last 2 hours
of sensor data.

CONCEPT (for interviews):
--------------------------
LSTM = Long Short-Term Memory — a type of Recurrent Neural Network
designed specifically for sequential/time-series data.

Problem with regular RNNs: vanishing gradient.
When backpropagating through 100+ timesteps, gradients shrink
to near-zero — early timesteps can't learn.

LSTM solution: cell state (Ct) — a "conveyor belt" of memory
that runs through the sequence with only minor linear interactions.
Information can be added or removed via gates:

1. Forget gate: ft = σ(Wf·[ht-1, xt] + bf)
   Decides what to throw away from cell state
   
2. Input gate: it = σ(Wi·[ht-1, xt] + bi)
   C̃t = tanh(WC·[ht-1, xt] + bC)
   Decides what new info to add
   
3. Cell update: Ct = ft * Ct-1 + it * C̃t
   
4. Output gate: ot = σ(Wo·[ht-1, xt] + bo)
   ht = ot * tanh(Ct)

WHY LSTM FOR OCCUPANCY PREDICTION:
People follow patterns — arrive at 9am, lunch at 1pm, leave at 6pm.
LSTM can learn these temporal patterns from sensor history and
predict future occupancy 30-60 minutes ahead.

This allows PRE-COOLING: start AC 20 minutes before people arrive
→ room is already comfortable when they get there.

INPUT: Last 24 timesteps × 4 features = (24, 4) tensor
       Features: [temperature, humidity, pir_occupancy, co2_ppm]
       
OUTPUT: Occupancy probability for next 6 timesteps (30 minutes)
        Shape: (6,) — each value 0.0 to 1.0
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  LSTM Model Definition                                               #
# ------------------------------------------------------------------ #

class OccupancyLSTM(nn.Module):
    """
    LSTM network for occupancy prediction.
    
    Architecture:
    Input (24, 4) → LSTM(hidden=64, layers=2) → FC(64→32) → FC(32→6) → Sigmoid
    
    The LSTM processes the sequence and produces a hidden state.
    The final hidden state is passed through fully connected layers
    to produce 6 occupancy probability predictions.
    """
    
    def __init__(
        self,
        input_size: int = 4,      # features per timestep
        hidden_size: int = 64,    # LSTM hidden units
        num_layers: int = 2,      # stacked LSTM layers
        output_steps: int = 6,    # predict 6 steps ahead (30 mins)
        dropout: float = 0.2
    ):
        super(OccupancyLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # LSTM layer
        # batch_first=True: input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_steps)
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for stable training
        self.bn = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: (batch_size, seq_len=24, features=4)
        output shape: (batch_size, 6) — probabilities for next 6 steps
        """
        # LSTM forward pass
        # lstm_out: (batch, seq, hidden) — output at each timestep
        # (hn, cn): final hidden state and cell state
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use only the last timestep's output
        # last_out: (batch, hidden)
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Sigmoid for probability output [0, 1]
        return self.sigmoid(out)


# ------------------------------------------------------------------ #
#  Data Generation from Digital Twin                                   #
# ------------------------------------------------------------------ #

def generate_training_data(n_days: int = 30) -> tuple:
    """
    Generate training data by running the Digital Twin for n_days.
    
    Creates sequences of (past_sensors, future_occupancy) pairs.
    
    Returns:
        X: shape (N, 24, 4) — input sequences
        y: shape (N, 6) — target occupancy for next 30 mins
    """
    from digital_twin.thermal_model import DigitalTwin, RoomConfig
    
    print(f"Generating {n_days} days of training data...")
    
    twin = DigitalTwin(seed=42)
    
    all_temps, all_hums, all_occs, all_co2s = [], [], [], []
    
    # Collect raw sensor data
    for day in range(n_days):
        twin.reset()
        
        day_temps, day_hums, day_occs, day_co2s = [], [], [], []
        
        done = False
        action = 3  # always on at 22°C for data collection
        
        while not done:
            obs, reward, done, info = twin.step(action)
            day_temps.append(info['temp'])
            day_hums.append(info['humidity'])
            day_occs.append(float(info['occupants'] > 0))  # binary
            day_co2s.append(info['co2'])
        
        all_temps.extend(day_temps)
        all_hums.extend(day_hums)
        all_occs.extend(day_occs)
        all_co2s.extend(day_co2s)
    
    # Normalize features
    temps = np.array(all_temps)
    hums = np.array(all_hums)
    occs = np.array(all_occs)
    co2s = np.array(all_co2s)
    
    temps_norm = (temps - temps.mean()) / (temps.std() + 1e-8)
    hums_norm = (hums - hums.mean()) / (hums.std() + 1e-8)
    co2s_norm = (co2s - co2s.mean()) / (co2s.std() + 1e-8)
    
    # Stack into feature matrix: (timesteps, 4)
    features = np.stack([temps_norm, hums_norm, occs, co2s_norm], axis=1)
    
    # Create sliding window sequences
    seq_len = 24    # 24 timesteps of history (2 minutes at 5s intervals)
    pred_len = 6    # predict 6 timesteps ahead (30 seconds)
    
    X, y = [], []
    
    for i in range(len(features) - seq_len - pred_len):
        X.append(features[i:i + seq_len])
        y.append(occs[i + seq_len:i + seq_len + pred_len])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Generated {len(X)} training sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


# ------------------------------------------------------------------ #
#  PyTorch Dataset                                                     #
# ------------------------------------------------------------------ #

class OccupancyDataset(Dataset):
    """PyTorch Dataset for occupancy prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------ #
#  Training Function                                                   #
# ------------------------------------------------------------------ #

def train_lstm(
    n_days: int = 20,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = "lstm_occupancy.pt"
) -> OccupancyLSTM:
    """
    Train the LSTM occupancy predictor.
    
    Uses BCELoss (Binary Cross-Entropy) since output is probability.
    Adam optimizer with learning rate scheduling.
    """
    print("=" * 60)
    print("LSTM Occupancy Predictor Training")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # Generate data
    X, y = generate_training_data(n_days=n_days)
    
    # Split 80/20 train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    train_dataset = OccupancyDataset(X_train, y_train)
    val_dataset = OccupancyDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = OccupancyLSTM().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            # Gradient clipping for stable LSTM training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss', color='#2196F3')
    plt.plot(val_losses, label='Val Loss', color='#FF9800')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('LSTM Occupancy Predictor Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_training.png', dpi=150)
    
    print(f"\n✅ LSTM training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")
    
    return model


def predict_occupancy(
    model: OccupancyLSTM,
    recent_sensors: np.ndarray
) -> np.ndarray:
    """
    Predict occupancy for next 30 minutes.
    
    recent_sensors: shape (24, 4) — last 24 timesteps of sensor data
                    columns: [temp_norm, hum_norm, occupancy, co2_norm]
    
    Returns: array of 6 occupancy probabilities (next 30 minutes)
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(recent_sensors).unsqueeze(0)  # add batch dim
        probs = model(x).numpy().squeeze()
    return probs


if __name__ == "__main__":
    model = train_lstm(n_days=15, epochs=20)
    
    # Test prediction
    print("\nTesting prediction with random input...")
    test_input = np.random.randn(24, 4).astype(np.float32)
    probs = predict_occupancy(model, test_input)
    print(f"Predicted occupancy for next 30 mins: {probs}")
    print("(Values close to 1 = occupied, close to 0 = empty)")
