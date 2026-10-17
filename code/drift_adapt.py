"""Drift mitigation: incremental retraining on a small labeled window.

The temporal probe (7.11) shows that both the flat and the hierarchical models
degrade under chronological drift. Here we ask whether a small labeled sample
from the deployment period recovers performance. The timeline is split into
train (first 60%), adaptation (next 10%, assumed lightly labeled in deployment),
and test (last 30%). We compare models trained on the train window only with
models retrained (warm) on train + adaptation.

Usage (from code/):
    python drift_adapt.py
"""
import numpy as np
import pandas as pd
from config import IEC_TRAIN, IEC_TEST
from baseline import prep
from search_partition import sort_classes
from final_table import flat_hist, hier_hist, metrics

DROP = ["time", "timestamp", "record_id", "source_ip", "destination_ip"]
LABEL = "class"


def load_iec():
    df = pd.concat([pd.read_csv(IEC_TRAIN, low_memory=False),
                    pd.read_csv(IEC_TEST, low_memory=False)], ignore_index=True)
    return df.drop_duplicates(subset=[c for c in df.columns if c != LABEL]).reset_index(drop=True)


def report(tag, yte, pred, cls):
    a, m, w, worst = metrics(yte, pred, cls)
    print("  {:<28} acc={:.3f} macro-F1={:.3f} worst={:.3f}".format(tag, a, m, worst))
    return m, worst


df = load_iec().sort_values("time", kind="stable").reset_index(drop=True)
X, y = prep(df, LABEL, DROP)
n = len(df)
c1, c2 = int(0.6 * n), int(0.7 * n)
Xtr, ytr = X.iloc[:c1], y.iloc[:c1]
Xad, yad = X.iloc[c1:c2], y.iloc[c1:c2]
Xte, yte = X.iloc[c2:], y.iloc[c2:]
cls, _ = sort_classes(ytr)
print("rows: train {} | adapt {} | test {}".format(len(Xtr), len(Xad), len(Xte)))
print("train counts:", dict(ytr.value_counts()))
print("adapt counts:", dict(yad.value_counts()))
print("test  counts:", dict(yte.value_counts()))

print("\nno adaptation (train window only):")
report("flat", yte, flat_hist(Xtr, ytr, Xte, 1), cls)
report("hierarchy", yte, hier_hist(Xtr, ytr, Xte, cls, 1), cls)

Xtr2 = pd.concat([Xtr, Xad]).reset_index(drop=True)
ytr2 = pd.concat([ytr, yad]).reset_index(drop=True)
cls2, _ = sort_classes(ytr2)
print("\nwith adaptation (train + 10% labeled deployment window):")
report("flat", yte, flat_hist(Xtr2, ytr2, Xte, 1), cls2)
report("hierarchy", yte, hier_hist(Xtr2, ytr2, Xte, cls2, 1), cls2)


def coral(Xs, Xt):
    """Unsupervised CORAL: align source covariance/mean to the target."""
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + np.eye(d)
    Us, Ss, _ = np.linalg.svd(Cs)
    Ut, St, _ = np.linalg.svd(Ct)
    Ws = Us @ np.diag(1.0 / np.sqrt(Ss)) @ Us.T
    Wt = Ut @ np.diag(np.sqrt(St)) @ Ut.T
    Xs2 = (Xs.values - Xs.values.mean(0)) @ Ws @ Wt + Xt.values.mean(0)
    return pd.DataFrame(Xs2, columns=Xs.columns)


print("\nwith unsupervised CORAL alignment (train aligned to test):")
Xtr_c = coral(Xtr, Xte)
report("flat", yte, flat_hist(Xtr_c, ytr, Xte, 1), cls)
report("hierarchy", yte, hier_hist(Xtr_c, ytr, Xte, cls, 1), cls)

# Same 70/30 split as the paper's drift table, for a directly comparable row.
c = int(0.7 * n)
Xtr70, ytr70 = X.iloc[:c], y.iloc[:c]
Xte70, yte70 = X.iloc[c:], y.iloc[c:]
cls70, _ = sort_classes(ytr70)
print("\n70/30 split, no adaptation:")
report("flat", yte70, flat_hist(Xtr70, ytr70, Xte70, 1), cls70)
report("hierarchy", yte70, hier_hist(Xtr70, ytr70, Xte70, cls70, 1), cls70)
Xc70 = coral(Xtr70, Xte70)
print("70/30 split, CORAL-aligned:")
report("flat", yte70, flat_hist(Xc70, ytr70, Xte70, 1), cls70)
report("hierarchy", yte70, hier_hist(Xc70, ytr70, Xte70, cls70, 1), cls70)

