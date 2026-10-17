"""Fixed-IR tail-support sweep on UNSW-NB15.

The IR sweep (ir_sweep.py) varies IR at fixed tail support. Here we do the
converse: we subsample every class proportionally so that IR stays ~431 while the
rarest class shrinks (130 -> few). This tests the decision rule directly: does the
hierarchy's gain over loss re-weighting shrink with tail support and cross over?

Usage (from code/):
    python tail_sweep.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from config import UNSW, SEED
from baseline import prep, ID_COLS
from search_partition import sort_classes
from nn_hier import fixed_partition

LABEL = "attack_cat"
SEEDS = [1, 2, 3, 4, 5]
FACTORS = [1.0, 0.6, 0.4, 0.25, 0.15, 0.08]


def hist(**kw):
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=200, early_stopping=True, **kw)


def macro(yte, pred, cls):
    return f1_score(yte, pred, average="macro", zero_division=0)


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


def subsample_prop(d, f, seed):
    parts = [g.sample(n=max(1, int(round(len(g) * f))), random_state=seed) for _, g in d.groupby(LABEL)]
    return pd.concat(parts).reset_index(drop=True)


df = pd.read_csv(UNSW, low_memory=False)
print("{:<7} {:>6} {:>10} {:>9} {:>9} {:>9} {:>10}".format("factor", "IR", "tailmin", "flat", "reweight", "hier", "hier-rw"))
rows = []
for f in FACTORS:
    d = subsample_prop(df, f, SEED)
    vc = d[LABEL].value_counts()
    IR = vc.max() / vc.min(); tm = int(vc.min())
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
    fr, rr, hr = np.mean(F), np.mean(R), np.mean(H)
    fs, rs, hs = np.std(F), np.std(R), np.std(H)
    rows.append((tm, IR, fr, rr, hr, hr - rr))
    print("{:<7} {:>6.0f} {:>10} {:>9} {:>9} {:>9} {:>+10.3f}".format(
        f, IR, tm,
        "{:.3f}+-{:.3f}".format(fr, fs),
        "{:.3f}+-{:.3f}".format(rr, rs),
        "{:.3f}+-{:.3f}".format(hr, hs), hr - rr))
print("\npaper (tailmin, hier, reweight, hier-rw):")
print("  " + " ".join("({},{:.3f},{:.3f},{:+.3f})".format(tm, h, r, g) for tm, IR, f, r, h, g in rows))
