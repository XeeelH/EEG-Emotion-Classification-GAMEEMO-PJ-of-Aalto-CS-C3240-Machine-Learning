from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from io_gameemo import iter_subject_game_csv
from features import bandpower_welch, BANDS


def build_dataset(root_dir: str, fs: int = 128, epoch_sec: int = 4):
    X, y = [], []
    for eeg, label, _ in iter_subject_game_csv(root_dir):
        C, T = eeg.shape
        L = fs * epoch_sec
        n = T // L
        for i in range(n):
            seg = eeg[:, i*L:(i+1)*L]
            feats = bandpower_welch(seg, fs)
            X.append(feats)
            y.append(label)
    X = np.stack(X)
    y = np.array(y)
    return X, y


def plot_confusion(y_true, y_pred, title="Confusion Matrix (MLP)"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["boring","calm","horror","funny"],
                yticklabels=["boring","calm","horror","funny"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.show()


def main():
    root = r"..\GAMEEMO"

    X, y = build_dataset(root, fs=128, epoch_sec=4)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
    mlp.fit(X_tr, y_tr)

    y_pred = mlp.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")

    print(f"MLP Accuracy: {acc:.4f}")
    print(f"MLP Macro-F1: {f1m:.4f}")
    print(classification_report(y_te, y_pred, digits=3))

    plot_confusion(y_te, y_pred, title="Confusion Matrix (MLP)")


if __name__ == "__main__":
    main()
