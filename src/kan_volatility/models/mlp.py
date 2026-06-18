from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class MLPRegressor(nn.Module):
    """
    Feedforward neural networks for volatility forecasting.

    The model takes engineered financial features and predicts log future 
    realized volatility.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        
        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class StandardScalerTorch:
    """
    Simple feature standardizer.

    We fit the mean and standard deviation on the training set only, then apply
    the same transformation to validation and test sets.
    """
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
    
    def fit(self, X: np.ndarray) -> "StandardScalerTorch":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def make_dataloader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
) -> DataLoader:
    """
    Convert NumPy arrays into a PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 30,
    random_state: int = 42,
    device: str | None = None,
) -> tuple[MLPRegressor, StandardScalerTorch, dict[str, list[float]]]:
    """
    Train an MLP with early stopping on validation loss.

    Returns
    -------
    model: Trained MLP model with the best validation weights loaded.
    scaler: Feature scaler fit on the training set.
    history: Dictionary containing train and validation loss curves.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    scaler = StandardScalerTorch()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_loader = make_dataloader(
        X=X_train_scaled,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_dataloader(
        X=X_val_scaled,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                loss = loss_fn(predictions, y_batch)

                val_losses.append(loss.item())
        
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_loss:.5f} | "
                f"val loss: {val_loss:.5f}"
            )
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state_dict)

    return model, scaler, history

def predict_mlp(
    model: MLPRegressor,
    scaler: StandardScalerTorch,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """
    Generate predictions from a trained MLP.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    return predictions

