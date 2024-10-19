from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import torch
import time
import numpy as np


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
