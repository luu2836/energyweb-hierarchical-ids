"""Fixed-IR tail-support sweep figure (UNSW-NB15).

Numbers from tail_sweep.py (5 seeds, mean +- std). IR stays ~431 while the rarest
class shrinks; the hierarchy's advantage over re-weighting shrinks with tail
support and crosses over near 40 samples.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

tail = np.array([130, 78, 52, 32, 20, 10])
flat = np.array([0.579, 0.567, 0.566, 0.554, 0.562, 0.554])
flat_s = np.array([0.005, 0.002, 0.009, 0.016, 0.022, 0.028])
rw = np.array([0.624, 0.609, 0.607, 0.607, 0.600, 0.593])
rw_s = np.array([0.001, 0.006, 0.006, 0.009, 0.008, 0.022])
hier = np.array([0.651, 0.623, 0.623, 0.603, 0.593, 0.569])
hier_s = np.array([0.013, 0.014, 0.006, 0.004, 0.023, 0.008])

fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
ax.axvspan(40, 200, color="#e8f0ff", zorder=0)
ax.axvspan(6, 40, color="#fff2e6", zorder=0)
ax.axvline(40, color="gray", ls="--", lw=1)
ax.errorbar(tail, flat, yerr=flat_s, marker="o", color="#888888", lw=1.6, capsize=3, label="flat GBDT")
ax.errorbar(tail, rw, yerr=rw_s, marker="s", color="#e07b1a", lw=2, capsize=3, label="re-weighting")
ax.errorbar(tail, hier, yerr=hier_s, marker="^", color="#1f5fbf", lw=2, capsize=3, label="hierarchy (ours)")
ax.set_xscale("log")
ax.set_xlabel("Tail support (rarest-class samples, log scale; IR fixed at ~431)")
ax.set_ylabel("Macro-F1")
ax.set_title("Hierarchy vs re-weighting as tail support shrinks")
ax.text(90, 0.655, "hierarchy\nregion", color="#1f5fbf", ha="center", fontsize=9)
ax.text(17, 0.548, "re-weighting\nregion", color="#e07b1a", ha="center", fontsize=9)
ax.set_ylim(0.54, 0.67)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "fig_tail_sweep.png")
fig.savefig(path)
print("wrote", path)
