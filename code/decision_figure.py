"""Decision figure: macro-F1 gain over the flat GBDT vs imbalance ratio.

Re-plots the macro-F1 means of Table 4 (comparison with imbalance-handling
methods) as a gain over the flat baseline, to expose the selection rule:
hierarchy at moderate imbalance, loss re-weighting at extreme imbalance.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IR = [24, 48, 431, 4545]
flat = [.9694, .9376, .5883, .7253]
hier = [.9805, .9570, .6551, .9022]
weight = [.9645, .9536, .6235, .9162]
smote = [.9661, .9564, .6491, .8925]
focal = [.9783, .9473, .6002, .8632]


def gain(m):
    return [a - b for a, b in zip(m, flat)]


fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
ax.axvspan(min(IR) * 0.75, 500, color="#e8f0ff", zorder=0)
ax.axvspan(500, max(IR) * 1.5, color="#fff2e6", zorder=0)
ax.axvline(1000, color="gray", ls="--", lw=1, zorder=1)
ax.axhline(0, color="k", lw=0.8, zorder=1)

ax.plot(IR, gain(hier), "o-", color="#1f5fbf", lw=2, label="hierarchy (ours)")
ax.plot(IR, gain(weight), "s-", color="#e07b1a", lw=2, label="re-weighting")
ax.plot(IR, gain(smote), "^-", color="#2e9e57", lw=1.6, label="SMOTE")
ax.plot(IR, gain(focal), "d--", color="#888888", lw=1.6, label="focal loss (MLP)")

ax.set_xscale("log")
ax.set_xlabel("Imbalance ratio (IR, log scale)")
ax.set_ylabel(r"$\Delta$ macro-F1 over flat-GBDT")
ax.set_title("When does structural decomposition help?")

ymax = max(gain(hier)) * 1.18
ax.set_ylim(min(gain(focal)) - 0.01, ymax)
ax.text(110, ymax * 0.9, "hierarchy\nregion", color="#1f5fbf", ha="center", fontsize=9)
ax.text(3000, ymax * 0.55, "re-weighting\nregion", color="#e07b1a", ha="center", fontsize=9)
for xi, yi in zip(IR, gain(hier)):
    ax.annotate("{:.3f}".format(yi), (xi, yi), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8, color="#1f5fbf")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "fig_decision.png")
fig.savefig(path)
print("wrote", path)
