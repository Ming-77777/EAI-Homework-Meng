import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


# -----------------------------
# Problem 5.1
# Decision Tree for Warehouse Hazard Prediction
# -----------------------------

out_dir = Path(__file__).resolve().parent / "Problem_5"
out_dir.mkdir(exist_ok=True)


def entropy_from_counts(pos: int, neg: int) -> float:
    total = pos + neg
    if total == 0:
        return 0.0
    ent = 0.0
    for count in (pos, neg):
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


def summarize_split(y: np.ndarray, mask: np.ndarray):
    y_left = y[mask]
    y_right = y[~mask]

    left_pos = int(y_left.sum())
    left_neg = len(y_left) - left_pos
    right_pos = int(y_right.sum())
    right_neg = len(y_right) - right_pos

    parent_pos = int(y.sum())
    parent_neg = len(y) - parent_pos

    parent_entropy = entropy_from_counts(parent_pos, parent_neg)
    left_entropy = entropy_from_counts(left_pos, left_neg)
    right_entropy = entropy_from_counts(right_pos, right_neg)

    weighted_child_entropy = (
        (len(y_left) / len(y)) * left_entropy
        + (len(y_right) / len(y)) * right_entropy
    )
    info_gain = parent_entropy - weighted_child_entropy

    return {
        "parent_pos": parent_pos,
        "parent_neg": parent_neg,
        "parent_entropy": parent_entropy,
        "left_n": len(y_left),
        "left_pos": left_pos,
        "left_neg": left_neg,
        "left_entropy": left_entropy,
        "right_n": len(y_right),
        "right_pos": right_pos,
        "right_neg": right_neg,
        "right_entropy": right_entropy,
        "weighted_child_entropy": weighted_child_entropy,
        "info_gain": info_gain,
    }


# Generate the dataset exactly as in the assignment
np.random.seed(42)
n = 300

load = np.random.randint(100, 1001, n)
inspection = np.random.randint(1, 91, n)
sensors = np.random.randint(1, 6, n)
floor_age = np.random.randint(1, 31, n)

true_risk = ((load > 500) | (inspection > 45)).astype(float)
flip = np.random.rand(n) < 0.20
high_risk = true_risk.copy()
high_risk[flip] = 1 - high_risk[flip]
high_risk = high_risk.astype(int)

df = pd.DataFrame(
    {
        "load_kg": load,
        "inspection_days": inspection,
        "sensors": sensors,
        "floor_age_years": floor_age,
        "high_risk": high_risk,
    }
)
df.to_csv(out_dir / "warehouse_hazard.csv", index=False)

X = df[["load_kg", "inspection_days", "sensors", "floor_age_years"]].to_numpy()
y = df["high_risk"].to_numpy()
feature_names = ["load_kg", "inspection_days", "sensors", "floor_age_years"]
class_names = ["low-risk", "high-risk"]

# -----------------------------
# Task 1: Information gain by hand
# -----------------------------
load_split = summarize_split(y, df["load_kg"].to_numpy() >= 500)
sensor_split = summarize_split(y, df["sensors"].to_numpy() <= 2)

print("=" * 72)
print("Task 1: Information Gain by Hand")
print("=" * 72)
print(
    f"Full dataset: n={len(y)}, high-risk={load_split['parent_pos']}, "
    f"low-risk={load_split['parent_neg']}"
)
print(f"Entropy(full dataset) = {load_split['parent_entropy']:.4f}\n")

print("Split on load_kg >= 500")
print(
    f"  >=500: {load_split['left_n']} examples "
    f"({load_split['left_pos']} high-risk, {load_split['left_neg']} low-risk), "
    f"entropy={load_split['left_entropy']:.4f}"
)
print(
    f"  <500 : {load_split['right_n']} examples "
    f"({load_split['right_pos']} high-risk, {load_split['right_neg']} low-risk), "
    f"entropy={load_split['right_entropy']:.4f}"
)
print(f"  Information gain = {load_split['info_gain']:.4f}\n")

print("Split on sensors <= 2")
print(
    f"  <=2 : {sensor_split['left_n']} examples "
    f"({sensor_split['left_pos']} high-risk, {sensor_split['left_neg']} low-risk), "
    f"entropy={sensor_split['left_entropy']:.4f}"
)
print(
    f"  >2  : {sensor_split['right_n']} examples "
    f"({sensor_split['right_pos']} high-risk, {sensor_split['right_neg']} low-risk), "
    f"entropy={sensor_split['right_entropy']:.4f}"
)
print(f"  Information gain = {sensor_split['info_gain']:.4f}\n")

better_root = "load_kg >= 500" if load_split["info_gain"] > sensor_split["info_gain"] else "sensors <= 2"
print(
    f"Better root split: {better_root} because it has the larger information gain "
    f"({max(load_split['info_gain'], sensor_split['info_gain']):.4f})."
)

# -----------------------------
# Task 2: First two levels
# -----------------------------
print("\n" + "=" * 72)
print("Task 2: First Two Levels")
print("=" * 72)

root_mask = df["load_kg"].to_numpy() >= 500
sensor_mask = df["sensors"].to_numpy() <= 2

branches = {
    "load>=500 AND sensors<=2": root_mask & sensor_mask,
    "load>=500 AND sensors>2": root_mask & ~sensor_mask,
    "load<500 AND sensors<=2": ~root_mask & sensor_mask,
    "load<500 AND sensors>2": ~root_mask & ~sensor_mask,
}

two_level_correct = 0
for name, mask in branches.items():
    branch_y = y[mask]
    pos = int(branch_y.sum())
    neg = len(branch_y) - pos
    pred = 1 if pos >= neg else 0
    pred_label = "high-risk" if pred == 1 else "low-risk"
    two_level_correct += max(pos, neg)
    print(f"{name}: {len(branch_y)} examples ({pos} high-risk, {neg} low-risk) -> predict {pred_label}")

two_level_acc = two_level_correct / len(y)
print(f"\nTwo-level tree training accuracy = {two_level_correct}/{len(y)} = {two_level_acc:.3f}")

print("\nResulting tree:")
print("Root: load_kg >= 500?")
print("|- Yes: sensors <= 2?")
print("|  |- Yes -> high-risk")
print("|  |- No  -> high-risk")
print("|- No: sensors <= 2?")
print("   |- Yes -> low-risk")
print("   |- No  -> low-risk")

# -----------------------------
# Tasks 3-5: sklearn implementation
# -----------------------------
print("\n" + "=" * 72)
print("Tasks 3-5: sklearn Implementation, Overfitting Analysis, Model Selection")
print("=" * 72)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"Training set: {len(X_train)} examples "
    f"({int(y_train.sum())} high-risk, {len(y_train) - int(y_train.sum())} low-risk)"
)
print(
    f"Test set: {len(X_test)} examples "
    f"({int(y_test.sum())} high-risk, {len(y_test) - int(y_test.sum())} low-risk)"
)

# Task 3: unlimited-depth tree
full_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
full_tree.fit(X_train, y_train)
full_train_acc = full_tree.score(X_train, y_train)
full_tree_text = export_text(
    full_tree,
    feature_names=feature_names,
    class_names=class_names,
)
(out_dir / "problem5_1_full_tree.txt").write_text(full_tree_text, encoding="utf-8")

print("\nUnlimited-depth tree training accuracy:", f"{full_train_acc:.3f}")
print("(Full tree text saved to problem5_1_full_tree.txt)")

# Task 4: overfitting analysis
print("\nOverfitting analysis:")
depths = [1, 2, 3, 4, 5, 6, None]
depth_labels = [str(d) if d is not None else "None" for d in depths]
train_accs = []
cv_means = []
cv_stds = []

for depth in depths:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train)
    cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring="accuracy")

    train_accs.append(train_acc)
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())

    print(
        f"max_depth={str(depth):>4} | train_acc={train_acc:.3f} | "
        f"cv_acc={cv_scores.mean():.3f} +- {cv_scores.std():.3f}"
    )

plt.figure(figsize=(8, 5))
plt.plot(depth_labels, train_accs, marker="o", label="Training accuracy")
plt.plot(depth_labels, cv_means, marker="s", label="CV accuracy")
cv_means_arr = np.array(cv_means)
cv_stds_arr = np.array(cv_stds)
plt.fill_between(
    range(len(depth_labels)),
    cv_means_arr - cv_stds_arr,
    cv_means_arr + cv_stds_arr,
    alpha=0.2,
)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Training vs. CV Accuracy")
plt.ylim(0.4, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "problem5_1_overfitting_curve.png", dpi=200)
plt.close()

# Task 5: model selection
best_idx = int(np.argmax(cv_means))
best_depth = depths[best_idx]
best_cv = cv_means[best_idx]

best_tree = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=42)
best_tree.fit(X_train, y_train)
best_tree_text = export_text(
    best_tree,
    feature_names=feature_names,
    class_names=class_names,
)
(out_dir / "problem5_1_best_tree.txt").write_text(best_tree_text, encoding="utf-8")

test_acc = best_tree.score(X_test, y_test)

print(f"\nBest depth by CV: {best_depth} (CV accuracy = {best_cv:.3f})")
print(f"Training accuracy at best depth: {best_tree.score(X_train, y_train):.3f}")
print(f"Test accuracy at best depth: {test_acc:.3f}")
print("(Best tree text saved to problem5_1_best_tree.txt)")

print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, best_tree.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.3f}")

# Save summary tables for easy submission / checking
summary_task1 = pd.DataFrame(
    [
        {
            "split": "load_kg >= 500",
            "left_n": load_split["left_n"],
            "left_high": load_split["left_pos"],
            "left_low": load_split["left_neg"],
            "left_entropy": round(load_split["left_entropy"], 4),
            "right_n": load_split["right_n"],
            "right_high": load_split["right_pos"],
            "right_low": load_split["right_neg"],
            "right_entropy": round(load_split["right_entropy"], 4),
            "information_gain": round(load_split["info_gain"], 4),
        },
        {
            "split": "sensors <= 2",
            "left_n": sensor_split["left_n"],
            "left_high": sensor_split["left_pos"],
            "left_low": sensor_split["left_neg"],
            "left_entropy": round(sensor_split["left_entropy"], 4),
            "right_n": sensor_split["right_n"],
            "right_high": sensor_split["right_pos"],
            "right_low": sensor_split["right_neg"],
            "right_entropy": round(sensor_split["right_entropy"], 4),
            "information_gain": round(sensor_split["info_gain"], 4),
        },
    ]
)
summary_task1.to_csv(out_dir / "problem5_1_task1_summary.csv", index=False)

summary_task45 = pd.DataFrame(
    {
        "max_depth": depth_labels,
        "train_accuracy": np.round(train_accs, 3),
        "cv_accuracy_mean": np.round(cv_means, 3),
        "cv_accuracy_std": np.round(cv_stds, 3),
    }
)
summary_task45.to_csv(out_dir / "problem5_1_depth_summary.csv", index=False)

print("\nFiles generated:")
for fname in [
    "warehouse_hazard.csv",
    "problem5_1_full_tree.txt",
    "problem5_1_best_tree.txt",
    "problem5_1_task1_summary.csv",
    "problem5_1_depth_summary.csv",
    "problem5_1_overfitting_curve.png",
]:
    print(f"  - {out_dir / fname}")
