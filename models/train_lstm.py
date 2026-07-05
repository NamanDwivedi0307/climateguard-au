import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import json
import os

# ── Device setup ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/features.csv")

FEATURE_COLS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "drought_factor", "days_since_rain", "ffdi",
    "fire_weather_composite", "month"
]
TARGET = "risk_score"

# ── Build sequences per postcode ───────────────────────────────────────────
# Group by postcode, sort by month to simulate time series
SEQ_LEN = 6  # 6 time steps (months) to predict next risk score

def build_sequences(df, seq_len=6):
    sequences, targets = [], []
    for postcode in df["postcode"].unique():
        sub = df[df["postcode"] == postcode].sort_values("month")
        X = sub[FEATURE_COLS].values
        y = sub[TARGET].values
        for i in range(len(X) - seq_len):
            sequences.append(X[i:i+seq_len])
            targets.append(y[i+seq_len])
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

print("Building time sequences...")
X_seq, y_seq = build_sequences(df, SEQ_LEN)
print(f"Sequences shape: {X_seq.shape} | Targets shape: {y_seq.shape}")

# Normalize features
X_mean = X_seq.mean(axis=(0,1))
X_std = X_seq.std(axis=(0,1)) + 1e-8
X_seq = (X_seq - X_mean) / X_std

y_mean, y_std = y_seq.mean(), y_seq.std()
y_seq_norm = (y_seq - y_mean) / y_std

# Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq_norm[:split], y_seq_norm[split:]

# ── Dataset ────────────────────────────────────────────────────────────────
class FireDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(FireDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(FireDataset(X_test, y_test), batch_size=64)

# ── LSTM Model ─────────────────────────────────────────────────────────────
class FireLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

model = FireLSTM(input_size=len(FEATURE_COLS)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── Training loop ──────────────────────────────────────────────────────────
print("\nTraining LSTM...")
EPOCHS = 30
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item()

    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/lstm_best.pth")

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ── Evaluate ───────────────────────────────────────────────────────────────
model.load_state_dict(torch.load("models/lstm_best.pth"))
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred = model(X_batch.to(device)).cpu().numpy()
        preds.extend(pred)
        actuals.extend(y_batch.numpy())

preds = np.array(preds) * y_std + y_mean
actuals = np.array(actuals) * y_std + y_mean
rmse = np.sqrt(((preds - actuals)**2).mean())
print(f"\nLSTM Test RMSE: {rmse:.2f}")

# Save normalisation params for inference
lstm_meta = {
    "X_mean": X_mean.tolist(),
    "X_std": X_std.tolist(),
    "y_mean": float(y_mean),
    "y_std": float(y_std),
    "seq_len": SEQ_LEN,
    "feature_cols": FEATURE_COLS,
    "rmse": round(rmse, 4)
}
with open("models/lstm_meta.json", "w") as f:
    json.dump(lstm_meta, f, indent=2)

print("LSTM model + metadata saved to models/")
