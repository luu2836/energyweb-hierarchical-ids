"""Decision figure: macro-F1 gain over the flat GBDT vs imbalance ratio.

Re-plots the macro-F1 means of Table 4 (aligned hierarchical values) as gains
over the flat baseline. The four benchmark datasets carry very different tail
supports, so the crossing is not a function of IR alone; each point is annotated
with its IR and its rarest-class count (Section 7.x, the IR and tail sweeps).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IR = [24, 48, 431, 4545]
tailmin = [1001, 1043, 130, 11]
name = ["Edge-IIoTset", "ToN-IoT", "UNSW-NB15", "CIC-IDS2017"]
flat = [.9694, .9376, .5883, .7253]
hier = [.9805, .9561, .6491, .9018]
weight = [.9645, .9536, .6235, .9162]


def gain(m):
    return [a - b for a, b in zip(m, flat)]


fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
ax.axhline(0, color="k", lw=0.8, zorder=1)
ax.plot(IR, gain(hier), "o-", color="#1f5fbf", lw=2, label="hierarchy (ours)")
ax.plot(IR, gain(weight), "s-", color="#e07b1a", lw=2, label="re-weighting")
ax.set_xscale("log")
ax.set_xlabel("Imbalance ratio (IR, log scale)")
ax.set_ylabel(r"$\Delta$ macro-F1 over flat-GBDT")
ax.set_title("Structural decomposition vs loss re-weighting by dataset")

for xi, ti, ni, hi, wi in zip(IR, tailmin, name, gain(hier), gain(weight)):
    ax.annotate("{}  (IR={}, min={})".format(ni, xi, ti), (xi, hi),
                textcoords="offset points", xytext=(6, 6), fontsize=7.5, color="#1f5fbf")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "fig_decision.png")
fig.savefig(path)
print("wrote", path)
