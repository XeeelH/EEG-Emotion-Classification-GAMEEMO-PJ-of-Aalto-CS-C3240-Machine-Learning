from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from io_gameemo import iter_subject_game_csv, EMOTIV_14
from features import bandpower_welch, BANDS


def build_dataset(root_dir: str, fs: int = 128, epoch_sec: int = 4):
    X, y = [], []
    for eeg, label, _ in iter_subject_game_csv(root_dir):
        C, T = eeg.shape
        L = fs * epoch_sec
        n = T // L
        for i in range(n):
            seg = eeg[:, i * L:(i + 1) * L]
            feats = bandpower_welch(seg, fs)
            X.append(feats)
            y.append(label)
    X = np.stack(X)
    y = np.array(y)
    return X, y


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["boring", "calm", "horror", "funny"],
                yticklabels=["boring", "calm", "horror", "funny"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: RandomForestClassifier, top_n: int = 20):
    channel_names = sorted(list(EMOTIV_14))
    band_names = list(BANDS.keys())
    feature_names = [f"{ch}_{band}" for ch in channel_names for band in band_names]

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)

    top_features = forest_importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_features.plot.barh(ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_ylabel("Features")
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


def main():
    root = r"..\GAMEEMO"

    print("--- Building Dataset ---")
    X, y = build_dataset(root, fs=128, epoch_sec=4)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = scaler.transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    print("Dataset prepared, split, and scaled.\n")

    print("--- Training Random Forest Classifier ---")
    rf = RandomForestClassifier(
        criterion='gini',
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_tr_scaled, y_tr)
    print("Model training complete.\n")

    print("--- Evaluating Model Performance ---")
    y_pred_tr = rf.predict(X_tr_scaled)
    training_accuracy = accuracy_score(y_tr, y_pred_tr)
    training_error = 1.0 - training_accuracy
    print(f"Random Forest Training Accuracy: {training_accuracy:.4f}")
    print(f"Random Forest Training Error: {training_error:.4f}\n")

    y_pred_te = rf.predict(X_te_scaled)

    print("Random Forest Results:")
    print(classification_report(y_te, y_pred_te, digits=4))

    print(f"Random Forest test errors: {1-accuracy_score(y_te, y_pred_te):.4f}")

    plot_confusion(y_te, y_pred_te, title="Confusion Matrix (Random Forest)")

    plot_feature_importance(rf)


if __name__ == "__main__":
    main()