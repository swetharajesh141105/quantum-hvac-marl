import numpy as np
import torch
import torch.nn as nn
from data_gen import generate_day_profile

SEQ_LEN = 12
PREDICT_AHEAD = 6

def build_dataset(num_days=50):
    X, y = [], []
    for day in range(num_days):
        profile = generate_day_profile(seed=day)
        occupancy = np.array(profile["occupancy"])
        for i in range(len(occupancy) - SEQ_LEN - PREDICT_AHEAD):
            X.append(occupancy[i : i + SEQ_LEN])
            y.append(occupancy[i + SEQ_LEN + PREDICT_AHEAD - 1])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X, y

class OccupancyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.relu(self.fc1(h_n[-1]))
        return torch.sigmoid(self.fc2(out))

def train_and_save(epochs=100):
    X, y = build_dataset()
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = OccupancyLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val)
                acc = ((val_pred > 0.5).float() == y_val).float().mean()
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} val_loss={val_loss.item():.4f} val_acc={acc.item():.2f}")

    torch.save(model.state_dict(), "occupancy_lstm.pt")
    print("Saved occupancy_lstm.pt")
    return model

def predict_occupancy_30min(model, recent_occupancy_sequence):
    model.eval()
    seq = torch.tensor(recent_occupancy_sequence, dtype=torch.float32).reshape(1, SEQ_LEN, 1)
    with torch.no_grad():
        prob = model(seq).item()
    return prob > 0.5, prob

if __name__ == "__main__":
    model = train_and_save()
    dummy_seq = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    will_occupy, prob = predict_occupancy_30min(model, dummy_seq)
    print(f"Predicted occupied in 30 min: {will_occupy} (prob={prob:.2f})")