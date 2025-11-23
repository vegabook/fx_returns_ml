# =============================================================================
#  SAME-DAY CROSS-SECTIONAL PREDICTOR (19 assets → asset 20)
#  Minimal, clear, correct, and uses real batches (batch_size=64)
# =============================================================================
# colorscheme base16-atelier-lakeside-light light

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import IPython
import optuna

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

breakpoint()

# Convert to PyTorch tensors
X = torch.tensor(nX)        
y = torch.tensor(ny)       


# Step 3: Train/validation split (80% train, 20% validation)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]


# ================================ OPTUNA TUNING ================================

ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'elu': nn.ELU}

class SameDayPredictor(nn.Module):
    def __init__(self, input_size, layers, act, dropout=0.0, layernorm=False):
        super().__init__()
        act_fn = ACTIVATIONS[act]
        mods = []
        prev = input_size
        for size in layers:
            mods.append(nn.Linear(prev, size))
            if layernorm: mods.append(nn.LayerNorm(size))
            mods.append(act_fn())
            if dropout > 0: mods.append(nn.Dropout(dropout))
            prev = size
        mods.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def objective(trial):
    # Hyperparameters
    n_layers     = trial.suggest_int("layers", 1, 6)
    layers       = [trial.suggest_int(f"h{i}", 32, 512) for i in range(n_layers)]
    act          = trial.suggest_categorical("act", ["relu", "gelu", "elu"])
    dropout      = trial.suggest_float("dropout", 0.0, 0.4)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd           = trial.suggest_float("wd", 1e-7, 1e-4, log=True)
    layernorm    = trial.suggest_categorical("layernorm", [True, False])
    batch_size   = trial.suggest_categorical("batch", [4, 8, 16, 32, 64, 128, 256])

    # DataLoaders with trial batch size
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SameDayPredictor(X.shape[1], layers, act, dropout, layernorm).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.MSELoss()

    # Quick training (15 epochs is enough for tuning)
    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # Pruning: kill bad trials early
        trial.report(loss.item(), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Final validation score
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(device))
        val_loss = crit(val_pred, y_val.to(device)).item()

    return val_loss

# ================================ RUN STUDY ================================

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

n_trials = 2000
print(f"Starting Optuna study ({n_trials} trials)...")
study.optimize(objective, n_trials=n_trials)

print("\nBEST HYPERPARAMETERS")
print(study.best_trial.params)
print(f"Best validation MSE: {study.best_value:.8f}")


#BEST HYPERPARAMETERS
#{'layers': 3, 'h0': 238, 'h1': 417, 'h2': 287, 'act': 'relu', 'dropout': 0.2757999852387004, 'lr': 0.0008126057927070138, 'wd': 4.608000431080565e-06, 'layernorm': False, 'batch': 32}
#Best validation MSE: 0.00011192


