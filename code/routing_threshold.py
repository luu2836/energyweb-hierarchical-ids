"""Learnable (validation-tuned) routing threshold for the hierarchical cascade.

Baseline cascade routing is a plain argmax over {level classes, Other}. Here each
level k gets a confidence threshold tau_k: a sample is assigned a level class only
if the top class probability is >= tau_k, otherwise it is routed down as "Other".
tau_k is tuned on a validation split carved from the training data (greedy,
level by level) to maximise validation macro-F1. tau_k = 0 reproduces argmax
routing, so the two variants are directly comparable.

Usage (from code/):
    python routing_threshold.py flat     # flat vs hier(argmax) vs hier(learned tau)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

from config import EDGE_ML, TON_NET, CIC, UNSW, SEED
from baseline import prep, ID_COLS
from search_partition import sort_classes, build_interval_model
from nn_hier import fixed_partition
from final_table import flat_hist, metrics, SEEDS

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_cascade(models, intervals, cls, X, tau):
    n = len(X)
    pred = np.array([None] * n, dtype=object)
    rem = np.ones(n, dtype=bool)
    ip = np.arange(n)
    K = len(intervals)
    for k, (j, i) in enumerate(intervals):
        if not rem.any():
            break
        ridx = ip[rem]
        m = models[k]
        proba = m.predict_proba(X.iloc[ridx])
        classes = list(m.classes_)
        last = (k == K - 1)
        for a, idx in enumerate(ridx):
            p = proba[a]
            am = int(np.argmax(p))
            lab = classes[am]
            if last:
                if lab != "Other":
                    pred[idx] = lab
                else:
                    cand = [(p[t], classes[t]) for t in range(len(classes)) if classes[t] != "Other"]
                    pred[idx] = max(cand)[1] if cand else cls[-1]
                rem[idx] = False
            elif lab != "Other" and p[am] >= tau[k]:
                pred[idx] = lab
                rem[idx] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


def tune_tau(models, intervals, cls, Xval, yval):
    K = len(intervals)
    tau = [0.0] * K
    best = metrics(yval, run_cascade(models, intervals, cls, Xval, tau), cls)[1]
    for k in range(K - 1):
        for c in GRID:
            trial = tau[:]
            trial[k] = c
            s = metrics(yval, run_cascade(models, intervals, cls, Xval, trial), cls)[1]
            if s > best + 1e-6:
                best, tau = s, trial
    return tau


def run(name, path, label_col, dedup):
    print("=" * 78)
    print(name, "| seeds", SEEDS)
    print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        df = df.drop_duplicates(subset=[c for c in df.columns if c != label_col]).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    res = {m: [] for m in ["flat", "hier-argmax", "hier-tau"]}
    taus = []
    for s in SEEDS:
        Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr_all = Xtr_all.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr_all = ytr_all.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr_all)
        cls_arr = np.array(cls, dtype=object)
        intervals = fixed_partition(len(cls), 4)
        # carve a validation split from the training data for threshold tuning
        Xtr, Xval, ytr, yval = train_test_split(Xtr_all, ytr_all, test_size=0.2, stratify=ytr_all, random_state=SEED)
        Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yval = yval.reset_index(drop=True)
        ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
        models_tune = [build_interval_model(Xtr, ypos, cls_arr, j, i, 200, 10 ** 9) for (j, i) in intervals]
        tau = tune_tau(models_tune, intervals, cls, Xval, yval)
        taus.append(tau)
        # final level models are retrained on the full training split (fair vs flat)
        ypos_all = pd.Series(ytr_all).map({c: k for k, c in enumerate(cls)}).values
        models = [build_interval_model(Xtr_all, ypos_all, cls_arr, j, i, 200, 10 ** 9) for (j, i) in intervals]

        res["flat"].append(metrics(yte, flat_hist(Xtr_all, ytr_all, Xte, s), cls))
        res["hier-argmax"].append(metrics(yte, run_cascade(models, intervals, cls, Xte, [0.0] * 4), cls))
        res["hier-tau"].append(metrics(yte, run_cascade(models, intervals, cls, Xte, tau), cls))
        print("  seed {} tau={}".format(s, tau))

    print("\n{:<14} {:>16} {:>17} {:>16}".format("method", "acc", "macro-F1", "worst-class-F1"))
    for m in res:
        a = np.array(res[m])
        print("{:<14} {:.4f}+-{:.4f}   {:.4f}+-{:.4f}    {:.4f}+-{:.4f}".format(
            m, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(), a[:, 3].mean(), a[:, 3].std()))
    a = np.array(res["hier-tau"])[:, 1]; b = np.array(res["hier-argmax"])[:, 1]
    _, pv = stats.ttest_rel(a, b)
    print("\nhier-tau vs hier-argmax macro-F1: diff={:+.4f}  p={:.4f}".format((a - b).mean(), pv))
    print("cells: flat {:.3f}/{:.3f} | argmax {:.3f}/{:.3f} | tau {:.3f}/{:.3f}".format(
        np.array(res["flat"])[:, 1].mean(), np.array(res["flat"])[:, 3].mean(),
        np.array(res["hier-argmax"])[:, 1].mean(), np.array(res["hier-argmax"])[:, 3].mean(),
        np.array(res["hier-tau"])[:, 1].mean(), np.array(res["hier-tau"])[:, 3].mean()))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    dss = {"edge": ("Edge-IIoTset", EDGE_ML, "Attack_type", False),
           "ton": ("ToN-IoT", TON_NET, "type", False),
           "unsw": ("UNSW-NB15", UNSW, "attack_cat", False),
           "cic": ("CIC-IDS2017", CIC, "Label", True)}
    if which in dss:
        run(*dss[which])
    else:
        for k in ["edge", "ton", "unsw", "cic"]:
            run(*dss[k])
