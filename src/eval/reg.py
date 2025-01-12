from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import torch
import time
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_logistic_regression(X, y, mode, do_norm):
    if do_norm:
        X = X / torch.norm(X, dim=1, keepdim=True)
    X, y = X.numpy(), y.numpy()
    prm = np.random.permutation(y.shape[0])
    X, y = X[prm, :], y[prm]
    
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    if mode == "self":
        model.fit(X, y)
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)
    elif mode == 'cv':
        accuracies = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        return np.mean(accuracies)
    else:
        assert False


def evaluate_catboost(X: torch.Tensor, y: torch.Tensor, do_norm):
    if do_norm:
        X = X / torch.norm(X, dim=1, keepdim=True)
    X_np = X.numpy()
    y_np = y.numpy()

    X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_val, label=y_val)

    model = CatBoostClassifier(task_type='GPU', iterations=500, verbose=100)

    model.fit(train_pool, early_stopping_rounds=50)

    y_val_pred = model.predict(val_pool)
    acc = accuracy_score(y_val, y_val_pred)
    return acc

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import torch


def evaluate_catboost_cv(X: torch.Tensor, y: torch.Tensor, num_steps: int, num_folds: int):
    # Convert PyTorch tensors to numpy arrays
    X_np = X.numpy()
    y_np = y.numpy()

    # Initialize K-Fold cross-validation with 3 folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    accuracies = []

    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        # Create CatBoost pools for training and validation
        train_pool = Pool(data=X_train, label=y_train)
        val_pool = Pool(data=X_val, label=y_val)

        # Initialize CatBoost model with configurable number of iterations
        model = CatBoostClassifier(task_type='GPU', iterations=num_steps, verbose=100)

        # Train the model
        model.fit(train_pool, early_stopping_rounds=50)

        # Predict and calculate accuracy for the current fold
        y_val_pred = model.predict(val_pool)
        acc = accuracy_score(y_val, y_val_pred)
        accuracies.append(acc)

        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

    # Calculate and print the mean accuracy across folds
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"Mean Accuracy: {mean_acc:.4f}")

    return mean_acc


def fft_extract_features(timeseries):
    """
    Performs FFT on the time series data and extracts frequency features.
    
    Args:
        timeseries (torch.Tensor): Input tensor of shape (batch_size, channels, time).
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 512) with frequency features.
    """
    batch_size, channels, time = timeseries.shape
    fft_result = torch.fft.rfft(timeseries, dim=-1)
    magnitude = torch.abs(fft_result)

    freq_features = torch.nn.functional.interpolate(
        magnitude,
        size=1024,  # Target size per channel
        mode="linear",
        align_corners=False
    )   # Shape: (batch_size, channels, 128)
    flattened_features = freq_features.view(batch_size, -1)  # Shape: (batch_size, channels * 128)
    return flattened_features