# =============================================================================
#  SAME-DAY CROSS-SECTIONAL PREDICTOR (19 assets → asset 20)
#  Minimal, clear, correct, and uses real batches (batch_size=64)
# =============================================================================
# colorscheme VIvid dark


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import IPython

# Step 1: Your data — shape (730, 20): 730 days × 20 return series
# Replace this line with your actual data
returns = np.random.randn(730, 20).astype(np.float32)   # ← YOUR real (730, 20) array here

with open("./data/crypto.parquet", "rb") as f:
    crypto = pl.read_parquet(f)
tickers = crypto.drop("date").columns


# drop date, log return, forward fill bad values, and drop first row of return NAs
returns = crypto.drop("date")\
                .with_columns(pl.all().pct_change().forward_fill())\
                .slice(1)\
                .to_numpy().astype(np.float32)

# target column by ticker, then make the dependent and indepentent variables
target_column = np.where(np.array(tickers) == "XBT")[0][0] 
ny = returns[:, target_column]         
nX = np.delete(returns, target_column, axis = 1)

# Convert to PyTorch tensors
X = torch.tensor(nX)
y = torch.tensor(ny)


# Step 3: Train/validation split (80% train, 20% validation)
split = int(0.75 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]


# Step 4: Wrap data in DataLoader with batch_size = 64 and shuffling
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64, shuffle=False)

# Step 5: The simplest possible neural network (MLP)
class SameDayPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 64),    # num columns inputs → 64 hidden units
            nn.ELU(),                    # activation
            nn.Linear(64, 32),            # 64 → 32
            nn.ELU(),
            nn.Linear(32, X.shape[1]),
            nn.ELU(),
            nn.Linear(X.shape[1], 1)              # 32 → 1 output (predicted return)
        )

    
    def forward(self, x):
        return self.net(x).squeeze(-1)   # output shape: (batch_size,)

# Step 6: Create model, loss function, and optimizer
model = SameDayPredictor()
criterion = nn.MSELoss()                 # Mean Squared Error = least squares
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Training loop — now using real batches of 64 days
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    # === Training phase ===
    model.train()               # enable dropout, batchnorm updates (none here, but good habit)
    train_loss_total = 0.0
    
    for batch_x, batch_y in train_loader:       # batch_x: (64, 19), batch_y: (64,)
        predictions = model(batch_x)            # forward pass
        loss = criterion(predictions, batch_y)  # compute MSE
        
        optimizer.zero_grad()      # clear old gradients
        loss.backward()            # compute new gradients (backpropagation)
        optimizer.step()           # update all weights
        
        train_loss_total += loss.item()
    
    avg_train_loss = train_loss_total / len(train_loader)

    # === Validation phase ===
    model.eval()                # disable dropout, etc.
    val_loss_total = 0.0
    with torch.no_grad():       # no gradients during validation → faster + saves memory
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            val_loss_total += loss.item()
    
    avg_val_loss = val_loss_total / len(val_loader)

    # Print progress
    print(f"Epoch {epoch:3d} | Train MSE: {avg_train_loss:.7f} | Val MSE: {avg_val_loss:.7f}")



model.eval()
with torch.no_grad():
    # Get ALL predictions for the validation set in one go (fastest way)
    all_predictions = model(X_val).cpu().numpy()      # shape (val_size,)
    all_actuals     = y_val.cpu().numpy()            # shape (val_size,)

# Convert to nice NumPy arrays
preds = all_predictions.flatten()
actuals = all_actuals.flatten()

IPython.embed()
