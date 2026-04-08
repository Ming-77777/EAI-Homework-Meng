from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold


# -------------------------------
# Generate the dataset
# -------------------------------
np.random.seed(0)
n = 50

distance = np.random.uniform(5, 40, n)      # meters
load = np.random.uniform(10, 100, n)        # kg
congestion = np.random.randint(0, 5, n)     # nearby robots

# True relationship (unknown to the model)
time = 1.8 * distance + 0.3 * load + 5.0 * congestion + 10 + np.random.normal(0, 5, n)

X = np.column_stack([distance, load, congestion])
y = time
feature_names = ["distance", "load", "congestion"]


# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} examples")
print(f"Test set:     {len(X_test)} examples")


# -------------------------------
# Task 1: Gradient descent
# -------------------------------
def gradient_descent(X, y, alpha=0.1, n_iter=1000, lam=0.0):
    """Linear regression by gradient descent with optional L2 regularization."""
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    N, d = X_norm.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for _ in range(n_iter):
        y_hat = X_norm @ w + b
        residuals = y - y_hat

        loss = np.mean(residuals**2) + lam * np.sum(w**2)
        losses.append(loss)

        grad_w = -2 / N * (X_norm.T @ residuals) + 2 * lam * w
        grad_b = -2 / N * np.sum(residuals)

        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, losses, X_mean, X_std


def predict(X_new, w, b, X_mean, X_std):
    X_norm = (X_new - X_mean) / X_std
    return X_norm @ w + b


def cv_mse(X, y, alpha=0.1, n_iter=1000, lam=0.0, k=5):
    """5-fold cross-validation MSE for gradient descent linear regression."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_mses = []

    for train_idx, val_idx in kf.split(X):
        w_cv, b_cv, _, Xm, Xs = gradient_descent(
            X[train_idx], y[train_idx], alpha=alpha, n_iter=n_iter, lam=lam
        )
        y_pred = predict(X[val_idx], w_cv, b_cv, Xm, Xs)
        fold_mses.append(np.mean((y[val_idx] - y_pred) ** 2))

    return np.mean(fold_mses), np.std(fold_mses)


w, b, losses, X_mean, X_std = gradient_descent(X_train, y_train, alpha=0.1, n_iter=1000)
cv_mean, cv_std = cv_mse(X_train, y_train, alpha=0.1, n_iter=1000)

print("\nTask 1 results")
print(f"Final training MSE: {losses[-1]:.2f}")
print(f"5-fold CV MSE:      {cv_mean:.2f} +/- {cv_std:.2f}")
print(f"Weights (normalized features): [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]")
print(f"Bias: {b:.2f}")


# -------------------------------
# Output directory for plots
# -------------------------------
out_dir = Path(__file__).resolve().parent / "Problem_5"
out_dir.mkdir(exist_ok=True)


# -------------------------------
# Task 2: Loss curve
# -------------------------------
plt.figure(figsize=(6.5, 4.2))
plt.plot(range(1, len(losses) + 1), losses, linewidth=2)
plt.title("Gradient Descent Convergence")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig(out_dir / "task2_loss_curve.png", dpi=200)
plt.close()

print("\nTask 2")
print("The MSE decreases monotonically and flattens near convergence.")


# -------------------------------
# Task 3: Learning rate experiments
# -------------------------------
alphas_to_plot = [0.001, 0.01, 0.1, 0.5]
plt.figure(figsize=(7.0, 4.6))

for alpha in alphas_to_plot:
    _, _, lr_losses, _, _ = gradient_descent(X_train, y_train, alpha=alpha, n_iter=500)
    plt.plot(range(1, len(lr_losses) + 1), lr_losses, linewidth=2, label=f"alpha = {alpha}")

plt.yscale("log")
plt.title("Learning Rate Comparison")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "task3_learning_rates.png", dpi=200)
plt.close()

_, _, losses_alpha_1, _, _ = gradient_descent(X_train, y_train, alpha=1.0, n_iter=500)

print("\nTask 3")
print("alpha = 0.001: converges very slowly")
print("alpha = 0.01: converges reliably")
print("alpha = 0.1: converges quickly and is a good choice")
print("alpha = 0.5: also converges quickly on this dataset")
print(f"alpha = 1.0: diverges (final loss = {losses_alpha_1[-1]:.2e})")
print("Approximate sweet spot: alpha around 0.1 (fast and stable).")
print("Too-large alpha overshoots the minimum and causes the updates to explode.")


# -------------------------------
# Task 4: L2 regularization
# -------------------------------
lambdas = [0, 0.01, 0.1, 1.0, 10.0]

print("\nTask 4")
print(f"{'lambda':>8} {'w_dist':>8} {'w_load':>8} {'w_cong':>8} {'Train MSE':>12} {'CV MSE':>12}")
print("-" * 62)

for lam in lambdas:
    # Slightly smaller alpha for very strong regularization.
    alpha_used = 0.01 if lam >= 10 else 0.1
    w_l2, b_l2, losses_l2, Xm_l2, Xs_l2 = gradient_descent(
        X_train, y_train, alpha=alpha_used, n_iter=1000, lam=lam
    )
    train_mse = losses_l2[-1] - lam * np.sum(w_l2**2)
    cv_mean_l2, cv_std_l2 = cv_mse(X_train, y_train, alpha=alpha_used, n_iter=1000, lam=lam)
    print(
        f"{lam:8.2f} {w_l2[0]:8.3f} {w_l2[1]:8.3f} {w_l2[2]:8.3f} "
        f"{train_mse:12.2f} {cv_mean_l2:12.2f}"
    )

print("As lambda increases, the weights shrink toward zero.")
print("At very large lambda, the model underfits and both training and CV MSE increase sharply.")


# -------------------------------
# Task 5: Logistic regression extension
# -------------------------------
def logistic_gd(X, y, alpha=0.1, n_iter=1000):
    """Logistic regression trained with gradient descent."""
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    N, d = X_norm.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    accuracies = []

    for _ in range(n_iter):
        z = X_norm @ w + b
        y_hat = 1 / (1 + np.exp(-z))

        eps = 1e-15
        y_hat_clip = np.clip(y_hat, eps, 1 - eps)
        loss = -np.mean(y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip))
        losses.append(loss)

        preds = (y_hat >= 0.5).astype(int)
        accuracies.append(np.mean(preds == y))

        grad_w = -1 / N * (X_norm.T @ (y - y_hat))
        grad_b = -1 / N * np.sum(y - y_hat)

        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, losses, accuracies, X_mean, X_std


# Above-median retrieval time = 1 ("slow")
y_class_train = (y_train > np.median(y_train)).astype(int)
y_class_test = (y_test > np.median(y_train)).astype(int)

w_log, b_log, log_losses, log_accs, Xm_log, Xs_log = logistic_gd(
    X_train, y_class_train, alpha=0.1, n_iter=1000
)

# Test accuracy
X_test_norm = (X_test - Xm_log) / Xs_log
test_probs = 1 / (1 + np.exp(-(X_test_norm @ w_log + b_log)))
test_pred = (test_probs >= 0.5).astype(int)
test_acc = np.mean(test_pred == y_class_test)

plt.figure(figsize=(6.5, 4.2))
plt.plot(range(1, len(log_losses) + 1), log_losses, linewidth=2)
plt.title("Logistic Regression: Loss")
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.tight_layout()
plt.savefig(out_dir / "task5_logistic_loss.png", dpi=200)
plt.close()

plt.figure(figsize=(6.5, 4.2))
plt.plot(range(1, len(log_accs) + 1), log_accs, linewidth=2)
plt.title("Logistic Regression: Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Training Accuracy")
plt.tight_layout()
plt.savefig(out_dir / "task5_logistic_accuracy.png", dpi=200)
plt.close()

print("\nTask 5")
print(f"Final cross-entropy: {log_losses[-1]:.4f}")
print(f"Training accuracy:   {log_accs[-1]:.3f}")
print(f"Test accuracy:       {test_acc:.3f}")


# -------------------------------
# Final evaluation on the held-out test set
# -------------------------------
y_test_pred_lin = predict(X_test, w, b, X_mean, X_std)
test_mse = np.mean((y_test - y_test_pred_lin) ** 2)

print("\nFinal evaluation on held-out test set")
print(f"Linear regression test MSE:      {test_mse:.2f}")
print(f"Logistic regression test acc.:   {test_acc:.3f}")
