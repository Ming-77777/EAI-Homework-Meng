from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class ArchConfig:
    name: str
    conv_channels: tuple[int, ...]
    use_batch_norm: bool = False


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    batch_size: int
    dropout: float = 0.0
    weight_decay: float = 0.0
    use_augmentation: bool = False
    early_stopping_patience: int | None = None


class ShelfDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.images[idx]).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ShelfCNN(nn.Module):
    def __init__(self, arch: ArchConfig, dropout: float = 0.0, num_classes: int = 3) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in arch.conv_channels:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if arch.use_batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        # Infer flattened size from architecture dynamically.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))

        classifier: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            classifier.append(nn.Dropout(p=dropout))
        classifier.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_shelf_images(n_per_class: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    size = 64
    class_names = ["normal", "damaged", "overloaded"]

    def _base_canvas() -> tuple[np.ndarray, int]:
        img = np.full((size, size), 0.10, dtype=np.float32)
        shelf_top = int(47 + rng.integers(-4, 5))
        shelf_h = 5
        img[shelf_top : shelf_top + shelf_h, :] = 0.75
        return img, shelf_top

    def _draw_boxes(img: np.ndarray, shelf_top: int, low_count: int, high_count: int, max_h: int) -> None:
        n = int(rng.integers(low_count, high_count + 1))
        for _ in range(n):
            w = int(rng.integers(4, 14))
            h = int(rng.integers(4, max_h + 1))
            x0 = int(rng.integers(0, max(1, size - w)))
            y1 = shelf_top
            y0 = max(4, y1 - h)
            intensity = float(rng.uniform(0.28, 0.55))
            img[y0:y1, x0 : x0 + w] = intensity

    def _draw_crack(img: np.ndarray, shelf_top: int) -> None:
        shelf_bottom = shelf_top + 5
        x = int(rng.integers(8, size - 8))
        y = int(rng.integers(shelf_top + 1, shelf_bottom - 1))
        slope = float(rng.uniform(-0.35, 0.35))
        length = int(rng.integers(10, 28))
        for step in range(length):
            xx = int(x + step)
            yy = int(y + step * slope)
            if 0 <= xx < size and shelf_top <= yy < shelf_bottom:
                img[yy, xx] = float(rng.uniform(0.0, 0.16))
                if yy + 1 < shelf_bottom:
                    img[yy + 1, xx] = float(rng.uniform(0.0, 0.16))

    def _noise(img: np.ndarray) -> np.ndarray:
        img = img + float(rng.uniform(-0.04, 0.04))
        img = img + rng.normal(0.0, 0.04, img.shape)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    images: list[np.ndarray] = []
    labels: list[int] = []

    for cls in range(3):
        for _ in range(n_per_class):
            img, shelf_top = _base_canvas()
            if cls == 0:
                _draw_boxes(img, shelf_top, low_count=2, high_count=5, max_h=15)
            elif cls == 1:
                _draw_boxes(img, shelf_top, low_count=2, high_count=5, max_h=15)
                _draw_crack(img, shelf_top)
            else:
                _draw_boxes(img, shelf_top, low_count=6, high_count=10, max_h=34)
                # Keep adding boxes until region looks dense.
                tries = 0
                while (img[:shelf_top] > 0.24).mean() < 0.45 and tries < 20:
                    _draw_boxes(img, shelf_top, low_count=2, high_count=4, max_h=30)
                    tries += 1

            images.append(_noise(img))
            labels.append(cls)

    images_np = np.stack(images, axis=0)
    labels_np = np.asarray(labels, dtype=np.int64)

    idx = rng.permutation(len(images_np))
    return images_np[idx], labels_np[idx], class_names


def load_or_create_data(data_path: Path, generate_if_missing: bool, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if data_path.exists():
        bundle = np.load(data_path)
        class_names = [str(x) for x in bundle["class_names"]]
        return bundle["images"], bundle["labels"], class_names

    if not generate_if_missing:
        raise FileNotFoundError(
            f"Cannot find dataset at {data_path}. Use --generate-if-missing to create it."
        )

    images, labels, class_names = generate_shelf_images(seed=seed)
    np.savez(data_path, images=images, labels=labels, class_names=np.array(class_names))
    return images, labels, class_names


def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train : n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val :])

    return (
        rng.permutation(np.array(train_idx)),
        rng.permutation(np.array(val_idx)),
        rng.permutation(np.array(test_idx)),
    )


def make_transforms(use_augmentation: bool) -> tuple[transforms.Compose | None, transforms.Compose | None]:
    if not use_augmentation:
        return None, None

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2),
        ]
    )
    return train_tf, None


def build_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    use_augmentation: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = stratified_split(labels, 0.70, 0.15, seed)
    train_tf, eval_tf = make_transforms(use_augmentation)

    train_ds = ShelfDataset(images[train_idx], labels[train_idx], transform=train_tf)
    val_ds = ShelfDataset(images[val_idx], labels[val_idx], transform=eval_tf)
    test_ds = ShelfDataset(images[test_idx], labels[test_idx], transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))

            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.cpu().numpy())

    pred_arr = np.concatenate(all_pred)
    true_arr = np.concatenate(all_true)
    acc = float((pred_arr == true_arr).mean())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, acc, pred_arr, true_arr


def feature_map_sizes(conv_layers: tuple[int, ...], start_hw: int = 64) -> list[int]:
    sizes = [start_hw]
    current = start_hw
    for _ in conv_layers:
        current = current // 2  # max-pooling with kernel=2
        sizes.append(current)
    return sizes


def train_with_history(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, list[float]], dict[str, Any], nn.Module]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0

    # Epoch-0 evaluation before any gradient updates.
    train_loss0, train_acc0, _, _ = evaluate_model(model, train_loader, criterion, device)
    val_loss0, val_acc0, _, _ = evaluate_model(model, val_loader, criterion, device)
    history["train_loss"].append(train_loss0)
    history["train_acc"].append(train_acc0)
    history["val_loss"].append(val_loss0)
    history["val_acc"].append(val_acc0)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _, _ = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if cfg.early_stopping_patience is not None and patience_count >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": max(history["val_acc"]),
        "last_val_acc": history["val_acc"][-1],
    }
    return history, summary, model


def architecture_search(
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[ArchConfig, list[dict[str, Any]]]:
    candidates = [
        ArchConfig(name="conv2_16_32", conv_channels=(16, 32), use_batch_norm=False),
        ArchConfig(name="conv3_16_32_64", conv_channels=(16, 32, 64), use_batch_norm=False),
        ArchConfig(name="conv4_8_16_32_64_bn", conv_channels=(8, 16, 32, 64), use_batch_norm=True),
        ArchConfig(name="conv3_24_48_96_bn", conv_channels=(24, 48, 96), use_batch_norm=True),
    ]

    results: list[dict[str, Any]] = []

    for arch in candidates:
        train_loader, val_loader, _, _, _, _ = build_loaders(
            images, labels, batch_size=batch_size, seed=seed, use_augmentation=False
        )

        model = ShelfCNN(arch=arch, dropout=0.0).to(device)
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            dropout=0.0,
            weight_decay=0.0,
            use_augmentation=False,
            early_stopping_patience=None,
        )

        history, summary, _ = train_with_history(model, train_loader, val_loader, cfg, device)
        params = int(sum(p.numel() for p in model.parameters()))
        spatial = feature_map_sizes(arch.conv_channels)

        results.append(
            {
                "arch": asdict(arch),
                "params": params,
                "feature_sizes": spatial,
                "history": history,
                "summary": summary,
            }
        )

    best = max(results, key=lambda r: r["summary"]["best_val_acc"])
    return ArchConfig(**best["arch"]), results


def run_regularization_study(
    best_arch: ArchConfig,
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[dict[str, Any], dict[str, Any], nn.Module]:
    train_loader, val_loader, _, _, _, _ = build_loaders(
        images, labels, batch_size=batch_size, seed=seed, use_augmentation=False
    )
    baseline_model = ShelfCNN(best_arch, dropout=0.0).to(device)
    baseline_cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        dropout=0.0,
        weight_decay=0.0,
        use_augmentation=False,
        early_stopping_patience=None,
    )
    baseline_history, baseline_summary, baseline_model = train_with_history(
        baseline_model, train_loader, val_loader, baseline_cfg, device
    )

    reg_runs: list[dict[str, Any]] = []
    for p_drop in (0.25, 0.5):
        train_loader_aug, val_loader_aug, _, _, _, _ = build_loaders(
            images, labels, batch_size=batch_size, seed=seed, use_augmentation=True
        )
        reg_model = ShelfCNN(best_arch, dropout=p_drop).to(device)
        reg_cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            dropout=p_drop,
            weight_decay=1e-4,
            use_augmentation=True,
            early_stopping_patience=15,
        )
        reg_history, reg_summary, reg_model = train_with_history(
            reg_model, train_loader_aug, val_loader_aug, reg_cfg, device
        )
        reg_runs.append(
            {
                "dropout": p_drop,
                "history": reg_history,
                "summary": reg_summary,
                "model": reg_model,
            }
        )

    best_reg = max(reg_runs, key=lambda r: r["summary"]["best_val_acc"])
    return (
        {"history": baseline_history, "summary": baseline_summary, "model": baseline_model},
        best_reg,
        best_reg["model"],
    )


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, trues = evaluate_model(model, test_loader, criterion, device)
    report = classification_report(
        trues,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(trues, preds)

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "preds": preds.tolist(),
        "trues": trues.tolist(),
    }


def plot_training_curves(
    baseline_history: dict[str, list[float]],
    regularized_history: dict[str, list[float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(baseline_history["train_acc"], label="baseline train", color="#1b6ca8")
    axes[0].plot(baseline_history["val_acc"], label="baseline val", color="#1b6ca8", linestyle="--")
    axes[0].plot(regularized_history["train_acc"], label="regularized train", color="#e07a5f")
    axes[0].plot(regularized_history["val_acc"], label="regularized val", color="#e07a5f", linestyle="--")
    axes[0].set_title("Accuracy (with epoch 0)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)

    axes[1].plot(baseline_history["train_loss"], label="baseline train", color="#1b6ca8")
    axes[1].plot(baseline_history["val_loss"], label="baseline val", color="#1b6ca8", linestyle="--")
    axes[1].plot(regularized_history["train_loss"], label="regularized train", color="#e07a5f")
    axes[1].plot(regularized_history["val_loss"], label="regularized val", color="#e07a5f", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=25)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def visualize_first_layer_filters(model: ShelfCNN, out_path: Path) -> None:
    first_conv: nn.Conv2d | None = None
    for module in model.features:
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        return

    filters = first_conv.weight.detach().cpu().numpy()[:, 0, :, :]
    n_filters = filters.shape[0]
    cols = min(8, n_filters)
    rows = int(np.ceil(n_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes_arr = np.atleast_2d(axes)
    vmin, vmax = float(filters.min()), float(filters.max())

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes_arr[r, c]
        ax.axis("off")
        if i < n_filters:
            ax.imshow(filters[i], cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_title(f"k{i}", fontsize=8)

    fig.suptitle("First Conv Layer Filters", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def show_prediction_examples(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    out_path: Path,
    max_each: int = 5,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = torch.device("cpu")

    model.eval()
    images_all: list[np.ndarray] = []
    preds_all: list[int] = []
    trues_all: list[int] = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            images_all.extend(xb.squeeze(1).cpu().numpy())
            preds_all.extend(pred.tolist())
            trues_all.extend(yb.cpu().numpy().tolist())

    preds = np.array(preds_all)
    trues = np.array(trues_all)
    images = np.array(images_all)

    correct_idx = np.where(preds == trues)[0][:max_each]
    wrong_idx = np.where(preds != trues)[0][:max_each]
    chosen = np.concatenate([correct_idx, wrong_idx])

    if len(chosen) == 0:
        return

    cols = max_each
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)

    for i in range(cols):
        for r in range(rows):
            axes[r, i].axis("off")

    for i, idx in enumerate(correct_idx):
        axes[0, i].imshow(images[idx], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(
            f"T:{class_names[trues[idx]]}\nP:{class_names[preds[idx]]}",
            fontsize=8,
            color="#2d7f5e",
        )

    for i, idx in enumerate(wrong_idx):
        axes[1, i].imshow(images[idx], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(
            f"T:{class_names[trues[idx]]}\nP:{class_names[preds[idx]]}",
            fontsize=8,
            color="#b53a3a",
        )

    axes[0, 0].set_ylabel("Correct", fontsize=10)
    axes[1, 0].set_ylabel("Misclassified", fontsize=10)
    fig.suptitle("Prediction Examples", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_transfer_learning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
    resnet = resnet.to(device)

    # Stage 1: feature extraction.
    for p in resnet.parameters():
        p.requires_grad = False
    for p in resnet.conv1.parameters():
        p.requires_grad = True
    for p in resnet.fc.parameters():
        p.requires_grad = True

    cfg1 = TrainConfig(
        epochs=12,
        lr=1e-3,
        batch_size=train_loader.batch_size or 32,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda t: t.requires_grad, resnet.parameters()), lr=cfg1.lr, weight_decay=cfg1.weight_decay)

    def _fit_simple(epochs: int) -> None:
        for _ in range(epochs):
            resnet.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(resnet(xb), yb)
                loss.backward()
                optimizer.step()

    _fit_simple(cfg1.epochs)

    # Stage 2: fine-tuning with lower lr.
    for p in resnet.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(resnet.parameters(), lr=1e-4, weight_decay=1e-4)
    _fit_simple(8)

    metrics = evaluate_test_set(resnet, test_loader, class_names, device)
    with (out_dir / "transfer_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN shelf inspector with architecture and regularization study")
    parser.add_argument("--data-path", type=Path, default=Path("shelf_images.npz"))
    parser.add_argument("--generate-if-missing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("assignment_outputs/problem_6_2"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--quick", action="store_true", help="small run for smoke testing")
    return parser.parse_args()


def resolve_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    images, labels, class_names = load_or_create_data(args.data_path, args.generate_if_missing, args.seed)
    print(f"Dataset: images={images.shape}, labels={labels.shape}, classes={class_names}")

    if args.quick:
        epochs = 8
    else:
        epochs = args.epochs

    # Step 1 + 2: architecture comparison.
    t0 = time.time()
    best_arch, arch_results = architecture_search(
        images=images,
        labels=labels,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=epochs,
        lr=args.lr,
    )
    print(f"Best architecture: {best_arch}")

    with (out_dir / "architecture_results.json").open("w", encoding="utf-8") as f:
        json.dump(arch_results, f, indent=2)

    # Step 3: full regularization toolkit.
    baseline, regularized, best_reg_model = run_regularization_study(
        best_arch=best_arch,
        images=images,
        labels=labels,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=epochs,
        lr=args.lr,
    )

    with (out_dir / "regularization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_arch": asdict(best_arch),
                "baseline_summary": baseline["summary"],
                "regularized_best_dropout": regularized["dropout"],
                "regularized_summary": regularized["summary"],
            },
            f,
            indent=2,
        )

    plot_training_curves(
        baseline_history=baseline["history"],
        regularized_history=regularized["history"],
        out_path=out_dir / "train_val_curves.png",
    )

    # Step 4: final test evaluation.
    _, _, test_loader, _, _, _ = build_loaders(
        images, labels, batch_size=args.batch_size, seed=args.seed, use_augmentation=False
    )
    test_metrics = evaluate_test_set(best_reg_model, test_loader, class_names, device)
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Step 5: learned feature + example predictions.
    visualize_first_layer_filters(best_reg_model, out_dir / "first_layer_filters.png")
    show_prediction_examples(
        best_reg_model,
        test_loader,
        class_names,
        out_dir / "prediction_examples.png",
        max_each=5,
        device=device,
    )

    if args.run_transfer:
        train_loader, val_loader, test_loader, _, _, _ = build_loaders(
            images, labels, batch_size=args.batch_size, seed=args.seed, use_augmentation=True
        )
        transfer_metrics = run_transfer_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            out_dir=out_dir,
        )
        print(f"Transfer test accuracy: {transfer_metrics['test_acc']:.4f}")

    elapsed = time.time() - t0
    print(f"Regularized test accuracy: {test_metrics['test_acc']:.4f}")
    print(f"Finished in {elapsed:.1f}s. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
