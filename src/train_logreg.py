from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from io_gameemo import iter_subject_game_csv
from features import bandpower_welch, BANDS

from showResult import plot_band_distribution


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


def main():
    root = "../GAMEEMO"
    X, y = build_dataset(root)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(
        max_iter=500, multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print("Macro-F1:", f1_score(y_te, y_pred, average="macro"))
    print(classification_report(y_te, y_pred))

if __name__ == "__main__":
    main()
