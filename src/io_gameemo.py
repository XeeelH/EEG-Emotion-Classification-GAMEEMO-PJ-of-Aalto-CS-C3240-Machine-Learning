from __future__ import annotations

import glob
import os
import re

import pandas as pd
import numpy as np

LABEL_MAP = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}  # boring, calm, horror, funny
EMOTIV_14 = {"AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"}
GAME_PAT = re.compile(r".*G([1-4]).*", re.IGNORECASE)


def get_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EMOTIV_14 if c in df.columns]
    return df[cols]


def iter_subject_game_csv(root_dir: str):
    pat = os.path.join(root_dir, "*", "Preprocessed EEG Data", ".csv format", "*AllChannels.csv")
    for csv_path in sorted(glob.glob(pat)):
        fname = os.path.basename(csv_path)
        m = GAME_PAT.search(fname)
        g = f"G{m.group(1)}".upper()
        y = LABEL_MAP[g]

        df = pd.read_csv(csv_path)
        num_df = get_columns(df)
        arr = num_df.to_numpy(dtype=float).T
        yield arr, y, csv_path


