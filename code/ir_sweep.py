"""Synthetic imbalance-ratio sweep on CIC-IDS2017.

Varying the per-class cap gives dataset variants with different imbalance ratios
IR = n_max/n_min while the absolute tail support (min per-class count) stays
fixed. For each variant we measure macro-F1 (mean +- std over seeds) of a flat
GBDT, an inverse-frequency re-weighted GBDT, and the hierarchical GBDT, to test
whether the hierarchy-vs-re-weighting trade-off is governed by IR alone.

All base learners use early_stopping=True: HistGradientBoostingClassifier's
default ('auto') disables early stopping below 10k training rows and otherwise
collapses on this data, which would contaminate the small-IR end of the sweep.

Usage (from code/):
    python ir_sweep.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from config import CIC, SEED
from baseline import prep, ID_COLS
from search_partition import sort_classes
from nn_hier import fixed_partition

CAPS = [1200, 2000, 3000, 6000, 12000, 25000, 50000]
SEEDS = [1, 2, 3, 4, 5]
LABEL = "Label"


def hist(**kw):
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=200, early_stopping=True, **kw)


def macro(yte, pred, cls):
    return f1_score(yte, pred, average="macro", zero_division=0)


def cap_df(df, C, seed):
    return pd.concat([g.sample(n=min(len(g), C), random_state=seed)
                      for _, g in df.groupby(LABEL)]).reset_index(drop=True)


def reweight(Xtr, ytr, Xte):
    cnt = ytr.value_counts(); K = len(cnt)
    w = ytr.map(lambda c: len(ytr) / (K * cnt[c])).values
    return hist().fit(Xtr, ytr, sample_weight=w).predict(Xte)


def level_model(X, ypos, cls_arr, j, i):
    mask = ypos >= j
    p = ypos[mask]
    labs = np.full(len(p), "Other", dtype=object)
    sel = (p >= j) & (p <= i)
    labs[sel] = cls_arr[p[sel]]
    return hist().fit(X[mask], labs)


def hpredict(Xtr, ytr, Xte, cls):
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    intervals = fixed_partition(len(cls), 4)
    pred = np.array([None] * len(Xte), dtype=object)
    rem = np.ones(len(Xte), dtype=bool); ip = np.arange(len(Xte))
    for (j, i) in intervals:
        if not rem.any():
            break
        m = level_model(Xtr, ypos, cls_arr, j, i)
        ridx = ip[rem]; out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


df = pd.read_csv(CIC, low_memory=False)
print("{:<7} {:>6} {:>6} {:>15} {:>15} {:>15} {:>11} {:>11}".format(
    "cap", "IR", "tailmin", "flat", "reweight", "hier", "hier-rw", "rw-flat"))
rows = []
for C in CAPS:
    d = cap_df(df, C, SEED).drop_duplicates().reset_index(drop=True)
    vc = d[LABEL].value_counts()
    IR = vc.max() / vc.min(); tailmin = int(vc.min())
    X, y = prep(d, LABEL, ID_COLS)
    F, R, H = [], [], []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr)
        F.append(macro(yte, hist().fit(Xtr, ytr).predict(Xte), cls))
        R.append(macro(yte, reweight(Xtr, ytr, Xte), cls))
        H.append(macro(yte, hpredict(Xtr, ytr, Xte, cls), cls))
    f, r, h = np.mean(F), np.mean(R), np.mean(H)
    fs, rs, hs = np.std(F), np.std(R), np.std(H)
    rows.append((IR, tailmin, f, r, h, fs, rs, hs))
    print("{:<7} {:>6.0f} {:>6} {:>7.3f}+-{:.3f} {:>7.3f}+-{:.3f} {:>7.3f}+-{:.3f} {:>+11.3f} {:>+11.3f}".format(
        C, IR, tailmin, f, fs, r, rs, h, hs, h - r, r - f))

print("\npaper (IR, tailmin, hier, reweight):")
print("  " + " ".join("({:.0f},{},{:.3f},{:.3f})".format(IR, tm, h, r) for IR, tm, f, r, h, *_ in rows))
