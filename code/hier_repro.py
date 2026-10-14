"""Recompute the hierarchy columns of Table 4 with the unified level-model seed.

Uses compare_all.cascade (level models seeded with config.SEED), so the hierarchy
matches Table 3 (final_table.py). Prints macro-F1 / worst-class F1 for the
hier and hier+weight rows.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import EDGE_ML, TON_NET, CIC, UNSW, SEED
from baseline import prep, ID_COLS
from search_partition import sort_classes
from compare_all import cascade, metrics

SEEDS = [1, 2, 3, 4, 5]
DS = [("Edge-IIoTset", EDGE_ML, "Attack_type", False),
      ("ToN-IoT", TON_NET, "type", False),
      ("UNSW-NB15", UNSW, "attack_cat", False),
      ("CIC-IDS2017", CIC, "Label", True)]

for name, path, lab, dedup in DS:
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        df = df.drop_duplicates(subset=[c for c in df.columns if c != lab]).reset_index(drop=True)
    X, y = prep(df, lab, ID_COLS)
    hier, hw = [], []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr); cls_arr = np.array(cls, dtype=object)
        ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
        hier.append(metrics(yte, cascade(Xtr, ypos, cls_arr, cls, Xte, SEED, False), cls))
        hw.append(metrics(yte, cascade(Xtr, ypos, cls_arr, cls, Xte, SEED, True), cls))
    a, b = np.array(hier), np.array(hw)
    print("{:<12} hier {:.4f}/{:.3f}   hier+weight {:.4f}/{:.3f}".format(
        name, a[:, 1].mean(), a[:, 3].mean(), b[:, 1].mean(), b[:, 3].mean()))
