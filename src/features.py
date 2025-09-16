from __future__ import annotations
import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}


def bandpower_welch(eeg_seg: np.ndarray, fs: int) -> np.ndarray:
    freqs, psd = welch(eeg_seg, fs=fs, nperseg=fs*2, axis=-1)
    feats = []
    for (fmin, fmax) in BANDS.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        bp = np.trapz(psd[:, idx], freqs[idx], axis=-1)  # [C]
        feats.append(bp)
    return np.concatenate(feats, axis=0)  # [C*len(BANDS)]


