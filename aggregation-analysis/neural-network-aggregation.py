import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import time
from datetime import datetime
import json
import seaborn as sns

# ====== Configuration ======
config = {
    # Data parameters
    "MAX_N_FRAMES": 40,

    # Model architecture options
    "MODEL_TYPE": "basic_ann",  # Options: basic_ann, residual, wide_deep, dense
    "HIDDEN_DIMS": [256, 128, 64, 32],  # Decreasing dimensions for each layer
    "DROPOUT_RATES": [0.4, 0.3, 0.2, 0.1],  # Tailored dropout for each layer
    "ACTIVATION": "silu",  # Options: relu, gelu, silu, leaky_relu

    # Training parameters
    "EPOCHS": 2000,
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY": 1e-3,
    "SEED": 42,

    # Feature engineering
    "FEATURE_SCALER": "minmax",  # Options: robust, standard, minmax
    "USE_STAT_FEATURES": True,  # Whether to use statistical features
    "USE_EXTENDED_STATS": True,  # Whether to use additional statistical features
    "USE_PERCENTILES": False,  # Whether to include percentiles
    "USE_HISTOGRAM": False,  # Whether to include histogram bins

    # Training strategy
    "EARLY_STOPPING_PATIENCE": 30,
    "GRADIENT_CLIPPING": 1.0,

    # Evaluation
    "METRICS": ["mse", "rmse", "mae", "r2", "pearson"],
}

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"plc_model_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Save config to file
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# ====== GPU Setup ======
print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(config["SEED"])
np.random.seed(config["SEED"])
if cuda_available:
    torch.cuda.manual_seed_all(config["SEED"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====== Read and Prepare Data ======
print("Loading and preparing data...")
df = pd.read_csv('results.csv')
df['PDBID'] = df['Pdbid_Frame'].apply(lambda x: x.split('_')[0])

# Analyze frame distribution
frame_counts = df.groupby('PDBID').size()
print(f"Frame statistics: min={frame_counts.min()}, max={frame_counts.max()}, mean={frame_counts.mean():.2f}")

# Group by PDB ID
grouped = df.groupby('PDBID')

X_train, y_train = [], []
X_test, y_test = [], []

for pdbid, group in grouped:
    preds = group['Predicted'].values
    real = group['Real'].values[0]

    # Frame statistics for this PLC
    if len(preds) > config["MAX_N_FRAMES"]:
        preds = preds[:config["MAX_N_FRAMES"]]
    elif len(preds) < config["MAX_N_FRAMES"]:
        # Zero padding
        preds = np.pad(preds, (0, config["MAX_N_FRAMES"] - len(preds)), 'constant', constant_values=0.0)

    # Feature engineering
    features = []

    # Raw predictions
    features.extend(preds)

    # Statistical features
    if config["USE_STAT_FEATURES"]:
        basic_stats = [
            np.min(preds),
            np.max(preds),
            np.mean(preds),
            np.std(preds),
        ]

        if config["USE_EXTENDED_STATS"]:
            extended_stats = [
                np.median(preds),
                np.mean(np.abs(preds - np.mean(preds))),  # MAD
                np.sqrt(np.mean(preds ** 2)),  # RMS
                (np.max(preds) - np.min(preds)),  # Range
            ]
            basic_stats.extend(extended_stats)

        if config["USE_PERCENTILES"]:
            percentiles = np.percentile(preds, [10, 25, 50, 75, 90])
            basic_stats.extend(percentiles)

        if config["USE_HISTOGRAM"]:
            hist, _ = np.histogram(preds, bins=5, range=(np.min(preds), np.max(preds)))
            basic_stats.extend(hist)

        features.extend(basic_stats)

    features = np.array(features, dtype=np.float32)

    if group['Set'].iloc[0].lower() == 'training':
        X_train.append(features)
        y_train.append(real)
    else:
        X_test.append(features)
        y_test.append(real)

# Convert to numpy arrays
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

# Update input dimension
INPUT_DIM = X_train.shape[1]
print(f"Input dimension after feature engineering: {INPUT_DIM}")

# Apply scaling
if config["FEATURE_SCALER"] == "robust":
    scaler = RobustScaler()
elif config["FEATURE_SCALER"] == "standard":
    scaler = StandardScaler()
else:  # minmax
    scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ====== PyTorch Dataset and Dataloader ======
class PLCDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== Model Architectures ======
def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "silu":
        return nn.SiLU()
    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    else:
        raise ValueError(f"Unknown activation: {activation_name}")

class BasicANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation="gelu"):
        super().__init__()
        layers = []
        prev_dim = input_dim

        activation_fn = get_activation(activation)

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class ResidualANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation="gelu"):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        activation_fn = get_activation(activation)

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation_fn,
                nn.Dropout(dropout_rate)
            )
            self.layers.append(block)

            # Skip connection if dimensions match
            if prev_dim == hidden_dim:
                self.layers.append(nn.Identity())
            else:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))

            prev_dim = hidden_dim

        self.final_layer = nn.Linear(prev_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(0, len(self.layers), 2):
            block = self.layers[i]
            skip = self.layers[i + 1]
            x = block(x) + skip(x)
        return self.final_layer(x)

class WideAndDeepANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation="gelu"):
        super().__init__()
        # Wide path (shallow)
        self.wide = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            get_activation(activation)
        )

        # Deep path
        deep_layers = []
        prev_dim = input_dim
        activation_fn = get_activation(activation)

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.LayerNorm(hidden_dim))
            deep_layers.append(activation_fn)
            deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.deep = nn.Sequential(*deep_layers)

        # Final combination
        self.combine = nn.Linear(hidden_dims[-1] * 2, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.combine(combined)

def create_model(model_type, input_dim, hidden_dims, dropout_rates, activation):
    if model_type == "basic_ann":
        return BasicANN(input_dim, hidden_dims, dropout_rates, activation)
    elif model_type == "residual":
        return ResidualANN(input_dim, hidden_dims, dropout_rates, activation)
    elif model_type == "wide_deep":
        return WideAndDeepANN(input_dim, hidden_dims, dropout_rates, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ====== Training Utilities ======
def create_logger(filename):
    """Create a simple logger that writes to a file"""
    with open(filename, 'w') as f:
        f.write("epoch,train_loss,lr\n")

    def log(epoch, train_loss, lr):
        with open(filename, 'a') as f:
            f.write(f"{epoch},{train_loss},{lr}\n")

    return log

def calculate_metrics(y_true, y_pred):
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    # Pearson correlation
    covariance = np.cov(y_true.flatten(), y_pred.flatten())[0, 1]
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true > 0 and std_pred > 0:
        metrics["pearson"] = covariance / (std_true * std_pred)
    else:
        metrics["pearson"] = 0.0

    return metrics

def train_model(model, train_loader, criterion, optimizer, scheduler,
                epochs, patience, log_func=None, grad_clip=None):
    train_losses = []
    best_train_loss = float('inf')
    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Adjust learning rate
        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log progress
        if log_func:
            log_func(epoch, train_loss, current_lr)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, LR: {current_lr:.2e}")

        # Early stopping check based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, train_losses, best_train_loss

def evaluate_model(model, X, y, dataset_name, output_dir, suffix=""):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    # Calculate metrics
    metrics = calculate_metrics(y, y_pred)

    # Create plot
    plt.figure(figsize=(10, 10))
    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=y.flatten(), y=y_pred.flatten(), alpha=0.7,
                         edgecolor='black', linewidth=0.5, s=80)

    # Add trend line (linear regression)
    z = np.polyfit(y.flatten(), y_pred.flatten(), 1)
    p = np.poly1d(z)
    x_vals = np.array(ax.get_xlim())
    ax.plot(x_vals, p(x_vals), "r--", linewidth=2,
            label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")

    # Add ideal line
    min_val = min(np.min(y), np.min(y_pred))
    max_val = max(np.max(y), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal')

    # Add metrics to plot
    textstr = '\n'.join([
        f'MSE: {metrics["mse"]:.4f}',
        f'RMSE: {metrics["rmse"]:.4f}',
        f'MAE: {metrics["mae"]:.4f}',
        f'R²: {metrics["r2"]:.4f}',
        f'Pearson: {metrics["pearson"]:.4f}'
    ])

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_xlabel("True Binding Affinity", fontsize=14)
    ax.set_ylabel("Predicted Binding Affinity", fontsize=14)
    ax.set_title(f"True vs Predicted ({dataset_name} Set)", fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(os.path.join(output_dir, f"prediction_{dataset_name.lower()}_{suffix}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    return metrics

def plot_training_curves(train_losses, output_dir, suffix=""):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Skip first few epochs for better visualization
    skip_epochs = min(4, len(train_losses) // 10)
    epochs_range = range(skip_epochs, len(train_losses))

    plt.plot(epochs_range, train_losses[skip_epochs:], label='Train Loss', color='blue', linewidth=2)

    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.title("Training Loss Curve", fontsize=16)
    plt.legend(fontsize=12)

    # Find minimum training loss and mark it
    min_train_idx = train_losses.index(min(train_losses))
    if min_train_idx >= skip_epochs:
        plt.scatter(min_train_idx, train_losses[min_train_idx], color='blue', s=100,
                    label=f'Best train loss: {train_losses[min_train_idx]:.6f}', zorder=5)
        plt.annotate(f'Epoch {min_train_idx}',
                     (min_train_idx, train_losses[min_train_idx]),
                     xytext=(10, -20),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    plt.savefig(os.path.join(output_dir, f"training_curve_{suffix}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

# ====== Main Execution ======
if __name__ == "__main__":
    start_time = time.time()

    print("\nTraining model on complete training dataset...")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        PLCDataset(X_train, y_train),
        batch_size=config["BATCH_SIZE"],
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        PLCDataset(X_test, y_test),
        batch_size=config["BATCH_SIZE"])

    # Initialize model
    model = create_model(
        config["MODEL_TYPE"],
        INPUT_DIM,
        config["HIDDEN_DIMS"],
        config["DROPOUT_RATES"],
        config["ACTIVATION"]
    ).to(device)

    # Initialize criterion, optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"])

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6)

    # Create logger
    log_func = create_logger(os.path.join(output_dir, "training_log.csv"))

    # Train the model
    model, train_losses, best_train_loss = train_model(
        model, train_loader, criterion, optimizer, scheduler,
        config["EPOCHS"], config["EARLY_STOPPING_PATIENCE"], log_func,
        grad_clip=config["GRADIENT_CLIPPING"])

    # Plot training curves
    plot_training_curves(train_losses, output_dir, "final")

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'config': config,
        'input_dim': INPUT_DIM,
    }, os.path.join(output_dir, "plc_model.pth"))

    # ====== Final Evaluation ======
    print("\nFinal Evaluation:")
    train_metrics = evaluate_model(model, X_train, y_train, "Train", output_dir, "final")
    test_metrics = evaluate_model(model, X_test, y_test, "Test", output_dir, "final")

    # Print summary of results
    print("\nSummary of Final Results:")
    print("Train Set:")
    for metric, value in train_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    print("\nTest Set:")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # Save results to file
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write("=== PLC Binding Affinity Model Results ===\n\n")

        f.write("Training Data Results:\n")
        for metric, value in train_metrics.items():
            f.write(f"{metric.upper()}: {value:.6f}\n")
        f.write("\n")

        f.write("Test Data Results:\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric.upper()}: {value:.6f}\n")

    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")
    print(f"All results saved to directory: {output_dir}")