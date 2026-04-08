from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Reproducibility helpers
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Data utilities
# -----------------------------

@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, x_train: np.ndarray, y_train: np.ndarray) -> "Standardizer":
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)

        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if y_std < 1e-8:
            y_std = 1.0

        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_std + self.y_mean


@dataclass
class FoldResult:
    fold_id: int
    val_rmse: float
    train_losses: List[float]
    val_losses: List[float]


@dataclass
class ExperimentBundle:
    cv_rmses: List[float]
    cv_mean_rmse: float
    cv_std_rmse: float
    fold_results: List[FoldResult]
    test_rmse_nn: float
    test_rmse_linear: float
    improvement_pct: float
    depth_results: Dict[str, Dict[str, float]]
    feature_names: List[str]


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_names = [str(name) for name in data["feature_names"]]
    return x, y, feature_names


def make_holdout_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split once using a fixed permutation, matching the assignment style."""
    n_samples = len(x)
    perm = np.random.default_rng(seed).permutation(n_samples)
    n_test = int(test_fraction * n_samples)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# -----------------------------
# Model and training
# -----------------------------

class FeedForwardRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y[:, None], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)



def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_val_t: torch.Tensor | None = None
    y_val_t: torch.Tensor | None = None
    if x_val is not None and y_val is not None:
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val[:, None], dtype=torch.float32)

    train_history: List[float] = []
    val_history: List[float] = []

    for _ in range(epochs):
        model.train()
        batch_losses: List[float] = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_history.append(float(np.mean(batch_losses)))

        if x_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(x_val_t), y_val_t).item())
            val_history.append(val_loss)

    return train_history, val_history



def predict_scaled(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32)
        return model(x_t).squeeze(1).cpu().numpy()



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def contiguous_folds(n_samples: int, n_folds: int) -> List[np.ndarray]:
    fold_size = n_samples // n_folds
    folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(n_folds)]
    if fold_size * n_folds < n_samples:
        folds[-1] = np.arange((n_folds - 1) * fold_size, n_samples)
    return folds



def cross_validate(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    hidden_dims: Sequence[int],
    n_folds: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
) -> Tuple[List[FoldResult], np.ndarray, np.ndarray]:
    folds = contiguous_folds(len(x_pool), n_folds)
    fold_results: List[FoldResult] = []
    train_curves: List[List[float]] = []
    val_curves: List[List[float]] = []

    for fold_id in range(n_folds):
        val_idx = folds[fold_id]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_id])

        scaler = Standardizer.fit(x_pool[train_idx], y_pool[train_idx])
        x_train = scaler.transform_x(x_pool[train_idx])
        y_train = scaler.transform_y(y_pool[train_idx])
        x_val = scaler.transform_x(x_pool[val_idx])
        y_val = scaler.transform_y(y_pool[val_idx])

        set_seed(seed + fold_id)
        model = FeedForwardRegressor(input_dim=x_pool.shape[1], hidden_dims=hidden_dims)
        loader = make_loader(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + fold_id,
        )
        train_history, val_history = run_training(
            model=model,
            train_loader=loader,
            x_val=x_val,
            y_val=y_val,
            epochs=epochs,
            lr=lr,
        )

        pred_val = scaler.inverse_y(predict_scaled(model, x_val))
        val_rmse = rmse(y_pool[val_idx], pred_val)

        fold_results.append(
            FoldResult(
                fold_id=fold_id + 1,
                val_rmse=val_rmse,
                train_losses=train_history,
                val_losses=val_history,
            )
        )
        train_curves.append(train_history)
        val_curves.append(val_history)

    return fold_results, np.array(train_curves), np.array(val_curves)



def fit_on_pool_and_test(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dims: Sequence[int],
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
) -> Dict[str, object]:
    scaler = Standardizer.fit(x_pool, y_pool)
    x_train = scaler.transform_x(x_pool)
    y_train = scaler.transform_y(y_pool)
    x_test_scaled = scaler.transform_x(x_test)

    set_seed(seed)
    model = FeedForwardRegressor(input_dim=x_pool.shape[1], hidden_dims=hidden_dims)
    loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    train_history, test_history = run_training(
        model=model,
        train_loader=loader,
        x_val=None,
        y_val=None,
        epochs=epochs,
        lr=lr,
    )

    nn_pred = scaler.inverse_y(predict_scaled(model, x_test_scaled))
    nn_rmse = rmse(y_test, nn_pred)

    x_train_aug = np.column_stack([x_train, np.ones(len(x_train), dtype=np.float32)])
    x_test_aug = np.column_stack([x_test_scaled, np.ones(len(x_test_scaled), dtype=np.float32)])
    weights = np.linalg.lstsq(x_train_aug, y_train, rcond=None)[0]
    linear_pred_scaled = x_test_aug @ weights
    linear_pred = scaler.inverse_y(linear_pred_scaled)
    linear_rmse = rmse(y_test, linear_pred)

    return {
        "model": model,
        "scaler": scaler,
        "train_history": train_history,
        "test_history": test_history,
        "nn_pred": nn_pred,
        "linear_pred": linear_pred,
        "nn_rmse": nn_rmse,
        "linear_rmse": linear_rmse,
    }



def compare_depths(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    n_folds: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    depth_map = {
        "1_hidden": [32],
        "2_hidden": [32, 16],
        "3_hidden": [32, 16, 8],
    }
    summary: Dict[str, Dict[str, float]] = {}
    mean_val_curves: Dict[str, np.ndarray] = {}

    for label, widths in depth_map.items():
        fold_results, _, val_curves = cross_validate(
            x_pool=x_pool,
            y_pool=y_pool,
            hidden_dims=widths,
            n_folds=n_folds,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        fold_rmses = np.array([item.val_rmse for item in fold_results], dtype=float)
        summary[label] = {
            "cv_rmse_mean": float(fold_rmses.mean()),
            "cv_rmse_std": float(fold_rmses.std()),
        }
        mean_val_curves[label] = val_curves.mean(axis=0)

    return summary, mean_val_curves


# -----------------------------
# Plotting
# -----------------------------


def plot_cv_curves(
    train_curves: np.ndarray,
    val_curves: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    epochs = np.arange(1, train_curves.shape[1] + 1)

    for curve in val_curves:
        ax.plot(epochs, curve, linewidth=1.0, alpha=0.35)

    ax.plot(epochs, train_curves.mean(axis=0), linewidth=2.2, linestyle="--", label="Mean train")
    ax.plot(epochs, val_curves.mean(axis=0), linewidth=2.4, label="Mean validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_title("Five-fold learning curves")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred_nn: np.ndarray,
    y_pred_linear: np.ndarray,
    nn_rmse: float,
    linear_rmse: float,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    lower = min(y_true.min(), y_pred_nn.min(), y_pred_linear.min()) - 2
    upper = max(y_true.max(), y_pred_nn.max(), y_pred_linear.max()) + 2

    panels = [
        (axes[0], y_pred_nn, f"Neural network\nRMSE = {nn_rmse:.2f} s"),
        (axes[1], y_pred_linear, f"Linear baseline\nRMSE = {linear_rmse:.2f} s"),
    ]
    for ax, preds, title in panels:
        ax.scatter(y_true, preds, s=22, alpha=0.7)
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Actual time (s)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Predicted time (s)")
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_depth_curves(mean_val_curves: Dict[str, np.ndarray], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    epochs = np.arange(1, len(next(iter(mean_val_curves.values()))) + 1)
    label_map = {
        "1_hidden": "1 hidden layer [32]",
        "2_hidden": "2 hidden layers [32, 16]",
        "3_hidden": "3 hidden layers [32, 16, 8]",
    }
    for key, curve in mean_val_curves.items():
        ax.plot(epochs, curve, linewidth=2.0, label=label_map[key])

    ax.set_title("Depth comparison on mean validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean validation MSE (normalized)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Orchestration
# -----------------------------


def run_experiment(data_path: str, output_dir: str) -> ExperimentBundle:
    os.makedirs(output_dir, exist_ok=True)
    torch.set_num_threads(1)
    if hasattr(torch, 'set_num_interop_threads'):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    x, y, feature_names = load_dataset(data_path)
    x_pool, x_test, y_pool, y_test = make_holdout_split(x, y, test_fraction=0.2, seed=0)

    fold_results, train_curves, val_curves = cross_validate(
        x_pool=x_pool,
        y_pool=y_pool,
        hidden_dims=[32, 16, 8],
        n_folds=5,
        batch_size=32,
        epochs=200,
        lr=1e-3,
        seed=0,
    )
    cv_rmses = [item.val_rmse for item in fold_results]

    final_run = fit_on_pool_and_test(
        x_pool=x_pool,
        y_pool=y_pool,
        x_test=x_test,
        y_test=y_test,
        hidden_dims=[32, 16, 8],
        batch_size=32,
        epochs=200,
        lr=1e-3,
        seed=0,
    )

    depth_summary, mean_val_curves = compare_depths(
        x_pool=x_pool,
        y_pool=y_pool,
        n_folds=5,
        batch_size=32,
        epochs=200,
        lr=1e-3,
        seed=0,
    )

    plot_cv_curves(train_curves, val_curves, os.path.join(output_dir, "cv_curves.png"))
    plot_predicted_vs_actual(
        y_true=y_test,
        y_pred_nn=final_run["nn_pred"],
        y_pred_linear=final_run["linear_pred"],
        nn_rmse=final_run["nn_rmse"],
        linear_rmse=final_run["linear_rmse"],
        out_path=os.path.join(output_dir, "predicted_vs_actual.png"),
    )
    plot_depth_curves(mean_val_curves, os.path.join(output_dir, "depth_experiment.png"))

    bundle = ExperimentBundle(
        cv_rmses=cv_rmses,
        cv_mean_rmse=float(np.mean(cv_rmses)),
        cv_std_rmse=float(np.std(cv_rmses)),
        fold_results=fold_results,
        test_rmse_nn=float(final_run["nn_rmse"]),
        test_rmse_linear=float(final_run["linear_rmse"]),
        improvement_pct=float((1.0 - final_run["nn_rmse"] / final_run["linear_rmse"]) * 100.0),
        depth_results=depth_summary,
        feature_names=feature_names,
    )

    metrics_path = os.path.join(output_dir, "metrics.json")
    payload = {
        "cv_rmses": bundle.cv_rmses,
        "cv_mean_rmse": bundle.cv_mean_rmse,
        "cv_std_rmse": bundle.cv_std_rmse,
        "fold_results": [asdict(item) for item in bundle.fold_results],
        "test_rmse_nn": bundle.test_rmse_nn,
        "test_rmse_linear": bundle.test_rmse_linear,
        "improvement_pct": bundle.improvement_pct,
        "depth_results": bundle.depth_results,
        "feature_names": bundle.feature_names,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Picking-Time Neural Network ===")
    print(f"Feature names: {bundle.feature_names}")
    print(f"CV RMSE (5-fold): {bundle.cv_mean_rmse:.2f} +/- {bundle.cv_std_rmse:.2f} sec")
    print(f"Final test RMSE - neural network: {bundle.test_rmse_nn:.2f} sec")
    print(f"Final test RMSE - linear baseline: {bundle.test_rmse_linear:.2f} sec")
    print(f"Relative test improvement: {bundle.improvement_pct:.1f}%")
    for key, stats in bundle.depth_results.items():
        print(
            f"{key}: CV RMSE = {stats['cv_rmse_mean']:.2f} +/- {stats['cv_rmse_std']:.2f} sec"
        )
    print(f"Saved figures and metrics to: {output_dir}")

    return bundle



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feedforward network for warehouse picking time.")
    parser.add_argument(
        "--data",
        default="picking_time_data.npz",
        help="Path to the .npz dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        default="assignment_outputs",
        help="Directory for figures and metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(data_path=args.data, output_dir=args.output_dir)
