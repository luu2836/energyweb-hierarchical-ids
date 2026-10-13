"""Energy-domain validation (flat vs hierarchical GBDT) on power-grid IDS data.

Runs the same leakage-aware protocol, the same fixed K=4 partition and the same
seeds as the IoT benchmarks, on two public power-grid datasets:

  * IEC61850  - Power System Intrusion Dataset (digital substation, IEC 61850 /
                GOOSE). Kaggle: sequincozes/power-system-intrusion-dataset.
  * PSCS      - Power System Cybersecurity Dataset (SCADA/PMU telemetry).
                Kaggle: colabsss/power-system-cybersecurity-dataset.

Usage (from code/):
    python energy_table.py iec         # concat + random stratified split
    python energy_table.py iec_split   # use the dataset's own train/test split
    python energy_table.py pscs
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

from config import IEC_TRAIN, IEC_TEST, PSCS
from baseline import prep
from search_partition import sort_classes, build_interval_model
from nn_hier import fixed_partition
from final_table import flat_hist, hier_hist, metrics, SEEDS

# Identifier / timestamp columns are removed (they are not traffic features and
# would otherwise leak record order).
DROP = ["time", "timestamp", "record_id", "source_ip", "destination_ip"]


def load_iec():
    tr = pd.read_csv(IEC_TRAIN, low_memory=False)
    te = pd.read_csv(IEC_TEST, low_memory=False)
    return pd.concat([tr, te], ignore_index=True)


def _dedup(df, label_col):
    feat = [c for c in df.columns if c != label_col]
    return df.drop_duplicates(subset=feat).reset_index(drop=True)


def _report(name, res):
    print("\n{:<14} {:>16} {:>17} {:>17} {:>16}".format(
        "method", "acc", "macro-F1", "weighted-F1", "worst-class-F1"))
    summary = {}
    for m in res:
        a = np.array(res[m])
        summary[m] = (a[:, 1].mean(), a[:, 3].mean())
        print("{:<14} {:.4f}+-{:.4f}   {:.4f}+-{:.4f}    {:.4f}+-{:.4f}    {:.4f}+-{:.4f}".format(
            m, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
            a[:, 2].mean(), a[:, 2].std(), a[:, 3].mean(), a[:, 3].std()))
    a = np.array(res["hier-HistGB"])[:, 1]
    b = np.array(res["flat-HistGB"])[:, 1]
    _, pv = stats.ttest_rel(a, b)
    print("\nhier vs flat macro-F1: diff={:+.4f}  paired t-test p={:.4f}".format((a - b).mean(), pv))
    print("paper row -> {}/{}  |  {}/{}".format(
        round(summary["flat-HistGB"][0], 3), round(summary["flat-HistGB"][1], 3),
        round(summary["hier-HistGB"][0], 3), round(summary["hier-HistGB"][1], 3)))
    print()


def run(name, df, label_col, dedup=True):
    print("=" * 78)
    print(name, "| rows", len(df), "| seeds", SEEDS)
    print("=" * 78)
    if dedup:
        df = _dedup(df, label_col)
        print("dedup rows:", len(df))

    X, y = prep(df, label_col, DROP)
    ir = y.value_counts()
    print("classes:", len(ir), "| IR(max/min) = {:.1f}".format(ir.max() / ir.min()))
    print("class counts:", dict(ir))

    res = {m: [] for m in ["flat-HistGB", "hier-HistGB"]}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr)
        res["flat-HistGB"].append(metrics(yte, flat_hist(Xtr, ytr, Xte, s), cls))
        res["hier-HistGB"].append(metrics(yte, hier_hist(Xtr, ytr, Xte, cls, s), cls))
        print("  seed {} done".format(s))
    _report(name, res)


def _predict_with_order(Xtr, ytr, Xte, cls_order, seed):
    """Run the frequency-ordered cascade with an externally given class order."""
    cls_arr = np.array(cls_order, dtype=object)
    pos = {c: k for k, c in enumerate(cls_order)}
    ypos = pd.Series(ytr).map(pos).values
    intervals = fixed_partition(len(cls_order), 4)
    n = len(Xte)
    pred = np.array([None] * n, dtype=object)
    rem = np.ones(n, dtype=bool)
    ip = np.arange(n)
    for (j, i) in intervals:
        if not rem.any():
            break
        m = build_interval_model(Xtr, ypos, cls_arr, j, i, 200, 10 ** 9)
        ridx = ip[rem]
        out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp
                rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls_order[-1]
    return pred


def run_reestimate(name, df, label_col, dedup=True, frac=0.7):
    """Does re-estimating the partition from deployment priors help?

    Temporal split (train on the first `frac` of the timeline). We compare three
    cascades that differ only in the class order used to build the partition:
      * train prior  - order from the training-period class frequencies (deployable)
      * deploy prior - order from the test-period class frequencies (oracle)
    If deploy-prior > train-prior, periodic re-estimation of the partition is
    worth it; if they are equal, re-estimation buys little.
    """
    print("=" * 78)
    print(name, "| partition re-estimation probe (temporal, first {:.0%})".format(frac))
    print("=" * 78)
    if dedup:
        df = _dedup(df, label_col)
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    X, y = prep(df, label_col, DROP)
    cut = int(len(df) * frac)
    Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
    Xte, yte = X.iloc[cut:], y.iloc[cut:]
    cls, _ = sort_classes(ytr)
    cls_te, _ = sort_classes(yte)
    order_oracle = list(cls_te) + [c for c in cls if c not in cls_te]
    print("train-prior order :", cls)
    print("deploy-prior order:", order_oracle)

    flat = metrics(yte, flat_hist(Xtr, ytr, Xte, 1), cls)
    hier_train = metrics(yte, _predict_with_order(Xtr, ytr, Xte, cls, 1), cls)
    hier_oracle = metrics(yte, _predict_with_order(Xtr, ytr, Xte, order_oracle, 1), cls)

    print("\n{:<38} {:>10} {:>11}".format("variant", "macro-F1", "worst-F1"))
    print("{:<38} {:.3f}      {:.3f}".format("flat-GBDT", flat[1], flat[3]))
    print("{:<38} {:.3f}      {:.3f}".format("hier, train-prior partition", hier_train[1], hier_train[3]))
    print("{:<38} {:.3f}      {:.3f}".format("hier, deploy-prior partition (oracle)", hier_oracle[1], hier_oracle[3]))
    print("paper row -> train-prior {:.3f}/{:.3f} | oracle {:.3f}/{:.3f}".format(
        round(hier_train[1], 3), round(hier_train[3], 3), round(hier_oracle[1], 3), round(hier_oracle[3], 3)))
    print()


def run_split(name, df_tr, df_te, label_col, dedup=True):
    print("=" * 78)
    print(name, "| own train/test split")
    print("=" * 78)
    if dedup:
        df_tr = _dedup(df_tr, label_col)
        df_te = _dedup(df_te, label_col)
    Xtr, ytr = prep(df_tr, label_col, DROP)
    Xte, yte = prep(df_te, label_col, DROP)
    print("train rows:", len(Xtr), "| test rows:", len(Xte))
    ir = ytr.value_counts()
    print("classes:", len(ir), "| IR(max/min) = {:.1f}".format(ir.max() / ir.min()))
    print("class counts:", dict(ir))
    cls, _ = sort_classes(ytr)
    for seed in SEEDS[:1]:
        flat = metrics(yte, flat_hist(Xtr, ytr, Xte, seed), cls)
        hier = metrics(yte, hier_hist(Xtr, ytr, Xte, cls, seed), cls)
    res = {"flat-HistGB": [flat], "hier-HistGB": [hier]}
    _report(name, res)


def run_drift(name, df, label_col, dedup=True, frac=0.7):
    """Random vs temporal split: robustness to prior drift.

    Sorts records by timestamp and trains on the first ``frac`` of the timeline,
    testing on the remainder. This deployment-drift proxy is harder (and more
    honest) than a random stratified split.
    """
    print("=" * 78)
    print(name, "| temporal drift probe (train on first {:.0%} of the timeline)".format(frac))
    print("=" * 78)
    if dedup:
        df = _dedup(df, label_col)
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    X, y = prep(df, label_col, DROP)
    cut = int(len(df) * frac)
    Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
    Xte, yte = X.iloc[cut:], y.iloc[cut:]
    cls, _ = sort_classes(ytr)
    missing = sorted(set(y.unique()) - set(cls))
    print("train rows:", len(Xtr), "| test rows:", len(Xte))
    print("train counts:", dict(ytr.value_counts()))
    print("test  counts:", dict(yte.value_counts()))
    if missing:
        print("NOTE: test classes unseen in train:", missing)

    t_flat = metrics(yte, flat_hist(Xtr, ytr, Xte, 1), cls)
    t_hier = metrics(yte, hier_hist(Xtr, ytr, Xte, cls, 1), cls)

    rf, rh = [], []
    for s in SEEDS:
        a, b, c_, d = train_test_split(X, y, test_size=1 - frac, stratify=y, random_state=s)
        a = a.reset_index(drop=True); b = b.reset_index(drop=True)
        c_ = c_.reset_index(drop=True); d = d.reset_index(drop=True)
        c2, _ = sort_classes(c_)
        rf.append(metrics(d, flat_hist(a, c_, b, s), c2))
        rh.append(metrics(d, hier_hist(a, c_, b, c2, s), c2))
    rf = np.array(rf); rh = np.array(rh)

    print("\n{:<26} {:>15} {:>15}".format("split", "flat macro/worst", "hier macro/worst"))
    print("{:<26} {:.3f}/{:.3f}   {:.3f}/{:.3f}".format(
        "random (5-seed mean)", rf[:, 1].mean(), rf[:, 3].mean(), rh[:, 1].mean(), rh[:, 3].mean()))
    print("{:<26} {:.3f}/{:.3f}   {:.3f}/{:.3f}".format(
        "temporal (first 70%)", t_flat[1], t_flat[3], t_hier[1], t_hier[3]))
    print("\ndegradation under drift (random -> temporal):")
    print("  flat  macro-F1 {:+.3f}   worst {:+.3f}".format(t_flat[1] - rf[:, 1].mean(), t_flat[3] - rf[:, 3].mean()))
    print("  hier  macro-F1 {:+.3f}   worst {:+.3f}".format(t_hier[1] - rh[:, 1].mean(), t_hier[3] - rh[:, 3].mean()))
    print("paper row -> temporal flat {:.3f}/{:.3f} | hier {:.3f}/{:.3f}".format(
        round(t_flat[1], 3), round(t_flat[3], 3), round(t_hier[1], 3), round(t_hier[3], 3)))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "iec"
    if which == "iec":
        run("IEC61850 (substation)", load_iec(), "class", dedup=True)
    elif which == "iec_split":
        run_split("IEC61850 (own split)",
                  pd.read_csv(IEC_TRAIN, low_memory=False),
                  pd.read_csv(IEC_TEST, low_memory=False), "class", dedup=True)
    elif which == "pscs":
        run("PowerSystemCS (SCADA/PMU)", pd.read_csv(PSCS, low_memory=False), "attack_label", dedup=True)
    elif which == "iec_drift":
        run_drift("IEC61850 (substation)", load_iec(), "class", dedup=True, frac=0.7)
    elif which == "iec_reestimate":
        run_reestimate("IEC61850 (substation)", load_iec(), "class", dedup=True, frac=0.7)
