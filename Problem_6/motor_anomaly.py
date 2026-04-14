from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunResult:
    name: str
    params: int
    train_loss: list[float]
    val_loss: list[float]
    val_acc: list[float]
    test_acc: float
    per_class_acc: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_motor_data(
    n_per_class: int = 400,
    seq_len: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate subtle motor-current anomalies for three classes."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8.0 * np.pi, seq_len)

    signals: list[np.ndarray] = []
    labels: list[int] = []
    class_names = ["healthy", "bearing_wear", "winding_fault"]

    for _ in range(n_per_class):
        amp = rng.uniform(0.85, 1.15)
        phase = rng.uniform(0, 2 * np.pi)
        load = 1.0 + rng.uniform(-0.12, 0.12) * np.sin(t * rng.uniform(0.08, 0.22))
        noise = rng.normal(0.0, rng.uniform(0.05, 0.09), size=seq_len)
        base = amp * load * np.sin(t + phase)
        signals.append((base + noise).astype(np.float32))
        labels.append(0)

    for _ in range(n_per_class):
        amp = rng.uniform(0.85, 1.15)
        phase = rng.uniform(0, 2 * np.pi)
        load = 1.0 + rng.uniform(-0.12, 0.12) * np.sin(t * rng.uniform(0.08, 0.22))
        noise = rng.normal(0.0, rng.uniform(0.05, 0.09), size=seq_len)
        # Subtle high-frequency ripple.
        ripple_freq = rng.uniform(14.0, 24.0)
        ripple_amp = rng.uniform(0.07, 0.18)
        ripple = ripple_amp * np.sin(ripple_freq * t + rng.uniform(0, 2 * np.pi))
        base = amp * load * np.sin(t + phase)
        signals.append((base + ripple + noise).astype(np.float32))
        labels.append(1)

    for _ in range(n_per_class):
        amp = rng.uniform(0.85, 1.15)
        phase = rng.uniform(0, 2 * np.pi)
        load = 1.0 + rng.uniform(-0.12, 0.12) * np.sin(t * rng.uniform(0.08, 0.22))
        noise = rng.normal(0.0, rng.uniform(0.05, 0.09), size=seq_len)
        base = amp * load * np.sin(t + phase)
        # Introduce asymmetric peaks via half-wave nonlinear term.
        asym = rng.uniform(0.16, 0.28)
        wave = base + asym * np.maximum(0.0, base)
        signals.append((wave + noise).astype(np.float32))
        labels.append(2)

    signals_np = np.asarray(signals, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)

    idx = rng.permutation(len(signals_np))
    return signals_np[idx], labels_np[idx], class_names


def load_or_create_dataset(
    data_path: Path,
    generate_if_missing: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if data_path.exists():
        obj = np.load(data_path)
        class_names = [str(x) for x in obj["class_names"]]
        return obj["sequences"], obj["labels"], class_names

    if not generate_if_missing:
        raise FileNotFoundError(
            f"Missing dataset at {data_path}. Use --generate-if-missing to create one."
        )

    seq, y, class_names = generate_motor_data(seed=seed)
    np.savez(data_path, sequences=seq, labels=y, class_names=np.array(class_names))
    return seq, y, class_names


def stratified_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []

    for cls in np.unique(labels):
        ids = np.where(labels == cls)[0]
        ids = rng.permutation(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    return (
        rng.permutation(np.asarray(train_ids)),
        rng.permutation(np.asarray(val_ids)),
        rng.permutation(np.asarray(test_ids)),
    )


def plot_waveforms(sequences: np.ndarray, labels: np.ndarray, class_names: list[str], out_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for cls in range(3):
        idx = int(np.where(labels == cls)[0][0])
        axes[cls].plot(sequences[idx], color="#2a6f9e", linewidth=1.0)
        axes[cls].set_title(class_names[cls])
        axes[cls].set_xlabel("time step")
        if cls == 0:
            axes[cls].set_ylabel("current")
        axes[cls].grid(alpha=0.2)
    fig.suptitle("One waveform per class (differences are subtle)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


class MotorLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class MotorCNN1D(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        feat = self.net(x).squeeze(-1)
        return self.head(feat)


class MotorTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 128,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        z = self.proj(x) + self.pos
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled)

    def first_layer_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights from first encoder layer: (B, H, T, T)."""
        z = self.proj(x) + self.pos
        first = self.encoder.layers[0]
        _, attn = first.self_attn(
            z,
            z,
            z,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = float((y_pred == y_true).mean())
    return float(np.mean(losses)), acc, y_pred, y_true


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    lr: float,
) -> RunResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_acc_hist: list[float] = []

    # Epoch 0 validation checkpoint.
    tr0, _, _, _ = evaluate(model, train_loader, criterion, device)
    vl0, va0, _, _ = evaluate(model, val_loader, criterion, device)
    train_loss_hist.append(tr0)
    val_loss_hist.append(vl0)
    val_acc_hist.append(va0)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        tr, _, _, _ = evaluate(model, train_loader, criterion, device)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        train_loss_hist.append(tr)
        val_loss_hist.append(vl)
        val_acc_hist.append(va)

        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, test_pred, test_true = evaluate(model, test_loader, criterion, device)
    per_class: dict[str, float] = {}
    for i, name in enumerate(class_names):
        m = test_true == i
        per_class[name] = float((test_pred[m] == test_true[m]).mean())

    return RunResult(
        name=model.__class__.__name__,
        params=int(sum(p.numel() for p in model.parameters())),
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        val_acc=val_acc_hist,
        test_acc=test_acc,
        per_class_acc=per_class,
    )


def plot_curves(results: list[RunResult], out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"MotorLSTM": "#2d6a4f", "MotorCNN1D": "#1d4e89", "MotorTransformer": "#b45309"}

    for r in results:
        c = colors.get(r.name, None)
        axes[0].plot(r.train_loss, label=f"{r.name} train", color=c)
        axes[0].plot(r.val_loss, linestyle="--", label=f"{r.name} val", color=c)
        axes[1].plot(r.val_acc, label=r.name, color=c)

    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def extract_attention_examples(
    model: MotorTransformer,
    x_test_seq: torch.Tensor,
    y_test: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    out_file: Path,
) -> None:
    model.eval()
    target_classes = [class_names.index("bearing_wear"), class_names.index("winding_fault")]

    idx_list: list[int] = []
    for cls in target_classes:
        ids = torch.where(y_test == cls)[0]
        if len(ids) > 0:
            idx_list.append(int(ids[0].item()))

    if len(idx_list) == 0:
        return

    x = x_test_seq[idx_list].to(device)
    with torch.no_grad():
        attn = model.first_layer_attention(x).cpu().numpy()  # (B, H, T, T)

    # Average over heads for cleaner visualization.
    attn = attn.mean(axis=1)

    fig, axes = plt.subplots(1, len(idx_list), figsize=(5.4 * len(idx_list), 4.6))
    axes = np.atleast_1d(axes)

    for i, idx in enumerate(idx_list):
        im = axes[i].imshow(attn[i], cmap="magma", aspect="auto")
        axes[i].set_title(f"Attention: {class_names[int(y_test[idx])]}\n(first encoder layer)")
        axes[i].set_xlabel("key time step")
        axes[i].set_ylabel("query time step")
        fig.colorbar(im, ax=axes[i], shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motor current anomaly classification with LSTM, 1D CNN, Transformer")
    parser.add_argument("--data-path", type=Path, default=Path("motor_current_data.npz"))
    parser.add_argument("--generate-if-missing", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("assignment_outputs/problem_6_3"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test run")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences, labels, class_names = load_or_create_dataset(
        data_path=args.data_path,
        generate_if_missing=args.generate_if_missing,
        seed=args.seed,
    )

    print(f"Device: {device}")
    print(f"Data: sequences={sequences.shape}, labels={labels.shape}, classes={class_names}")

    plot_waveforms(sequences, labels, class_names, out_dir / "waveforms_by_class.png")

    train_idx, val_idx, test_idx = stratified_indices(labels, seed=args.seed)
    x_train = sequences[train_idx]
    x_val = sequences[val_idx]
    x_test = sequences[test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    # Sequence-first tensors for LSTM and Transformer: (B, T, 1)
    x_train_seq = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    x_val_seq = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    x_test_seq = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)

    # Channel-first tensors for 1D CNN: (B, 1, T)
    x_train_cnn = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    x_val_cnn = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    x_test_cnn = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    seq_train_loader = DataLoader(TensorDataset(x_train_seq, y_train_t), batch_size=args.batch_size, shuffle=True)
    seq_val_loader = DataLoader(TensorDataset(x_val_seq, y_val_t), batch_size=args.batch_size, shuffle=False)
    seq_test_loader = DataLoader(TensorDataset(x_test_seq, y_test_t), batch_size=args.batch_size, shuffle=False)

    cnn_train_loader = DataLoader(TensorDataset(x_train_cnn, y_train_t), batch_size=args.batch_size, shuffle=True)
    cnn_val_loader = DataLoader(TensorDataset(x_val_cnn, y_val_t), batch_size=args.batch_size, shuffle=False)
    cnn_test_loader = DataLoader(TensorDataset(x_test_cnn, y_test_t), batch_size=args.batch_size, shuffle=False)

    epochs = 10 if args.quick else args.epochs

    lstm = MotorLSTM(hidden_size=32, num_classes=3).to(device)
    cnn = MotorCNN1D(num_classes=3).to(device)
    trf = MotorTransformer(seq_len=128, d_model=32, nhead=4, num_layers=2, ff_dim=64, num_classes=3).to(device)

    results: list[RunResult] = []
    results.append(
        train_model(
            lstm,
            seq_train_loader,
            seq_val_loader,
            seq_test_loader,
            class_names,
            device,
            epochs=epochs,
            lr=args.lr,
        )
    )
    results.append(
        train_model(
            cnn,
            cnn_train_loader,
            cnn_val_loader,
            cnn_test_loader,
            class_names,
            device,
            epochs=epochs,
            lr=args.lr,
        )
    )
    results.append(
        train_model(
            trf,
            seq_train_loader,
            seq_val_loader,
            seq_test_loader,
            class_names,
            device,
            epochs=epochs,
            lr=args.lr,
        )
    )

    plot_curves(results, out_dir / "train_val_curves.png")
    extract_attention_examples(
        trf,
        x_test_seq,
        y_test_t,
        class_names,
        device,
        out_dir / "transformer_attention_heatmaps.png",
    )

    summary: dict[str, Any] = {
        "data": {
            "n_total": int(len(sequences)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "class_names": class_names,
        },
        "results": [
            {
                "name": r.name,
                "params": r.params,
                "test_acc": r.test_acc,
                "per_class_acc": r.per_class_acc,
            }
            for r in results
        ],
    }

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nModel comparison on test set")
    for r in results:
        cls_txt = ", ".join([f"{k}:{v:.3f}" for k, v in r.per_class_acc.items()])
        print(f"- {r.name:16s} params={r.params:6d} test_acc={r.test_acc:.4f} | {cls_txt}")

    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
