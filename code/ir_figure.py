"""IR-sweep figure: macro-F1 vs imbalance ratio on CIC-IDS2017.

Numbers produced by ir_sweep.py (5 seeds, mean +- std). The tail classes hold a
fixed 11 samples across all caps, so the sweep changes IR only. Loss re-weighting
stays at or above the hierarchy across the whole spectrum, showing that the
imbalance ratio alone does not decide the winner.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IR = np.array([109, 182, 273, 545, 1091, 2273, 4545])
flat = np.array([0.794, 0.767, 0.763, 0.749, 0.749, 0.741, 0.729])
flat_s = np.array([0.039, 0.030, 0.020, 0.007, 0.009, 0.008, 0.004])
rw = np.array([0.922, 0.923, 0.910, 0.922, 0.924, 0.922, 0.913])
rw_s = np.array([0.010, 0.020, 0.011, 0.006, 0.010, 0.010, 0.007])
hier = np.array([0.912, 0.902, 0.908, 0.913, 0.912, 0.910, 0.901])
hier_s = np.array([0.013, 0.008, 0.009, 0.009, 0.011, 0.013, 0.008])

fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
ax.errorbar(IR, flat, yerr=flat_s, marker="o", color="#888888", lw=1.6, capsize=3, label="flat GBDT")
ax.errorbar(IR, rw, yerr=rw_s, marker="s", color="#e07b1a", lw=2, capsize=3, label="re-weighting")
ax.errorbar(IR, hier, yerr=hier_s, marker="^", color="#1f5fbf", lw=2, capsize=3, label="hierarchy (ours)")
ax.set_xscale("log")
ax.set_xlabel("Imbalance ratio (IR, log scale; tail support fixed at 11 samples)")
ax.set_ylabel("Macro-F1")
ax.set_title("Structural decomposition vs loss re-weighting across the IR spectrum")
ax.set_ylim(0.70, 0.95)
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "fig_ir_sweep.png")
fig.savefig(path)
print("wrote", path)
