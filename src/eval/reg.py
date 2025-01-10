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