"""Framework schematic for the leakage-aware hierarchical IDS (Figure 1)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

BLUE = "#1f5fbf"
LBLUE = "#e8f0ff"
LLBLUE = "#f3f8ff"
ORANGE = "#e07b1a"
GRAY = "#555555"


def box(x, y, w, h, text, fc="white", ec=BLUE, fs=8, weight="normal", tc="black", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, color=tc)


def arrow(x1, y1, x2, y2, color=GRAY, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=lw, ls=ls))


# --- top pipeline ---
box(1, 40, 19, 14, "Raw IIoT /\nsmart-grid\ntraffic flows", fs=8)
box(24, 38, 21, 17, "Leakage-aware\npreprocessing:\ndeduplicate flows,\ndrop label-leaking\nfeatures",
    fc=LBLUE, fs=7)
box(49, 38, 22, 17, "Frequency-ordered\npartition:\nsort classes by count,\nsplit into $K=4$\nlevels",
    fc=LBLUE, fs=7)
arrow(20, 47, 24, 47)
arrow(45, 46, 49, 46)

# --- cascade container ---
ax.add_patch(FancyBboxPatch((1, 13), 74, 21, boxstyle="round,pad=0.5,rounding_size=1.8",
                            fc=LLBLUE, ec=BLUE, lw=1.5))
ax.text(38, 31.2, "Hierarchical divide-and-conquer cascade", ha="center", va="center",
        fontsize=8.8, weight="bold", color=BLUE)

levels = ["Level 1\nhead\n+ Other", "Level 2", ". . .", "Level $K$\ntail"]
w, gap, x0 = 16, 2, 3
for i, t in enumerate(levels):
    xi = x0 + i * (w + gap)
    box(xi, 16, w, 10, t, fc="white", fs=7.6,
        weight="bold" if i in (0, 3) else "normal")
    if i < len(levels) - 1:
        arrow(xi + w, 21, xi + w + gap, 21, color=ORANGE, lw=1.1)
ax.text(41, 14.4, "unresolved samples routed as ``Other'' to the next level",
        fontsize=6.4, color=ORANGE, ha="center")

arrow(60, 38, 60, 34)

# --- outputs ---
box(1, 2, 32, 9, "Predicted class\n(normal or attack family)", fs=7.6)
box(36, 2, 39, 9, "Base learner: gradient-boosted trees  |  gated MLP", fs=7.2)
arrow(16, 13, 16, 11)
arrow(55, 13, 55, 11)

# --- edge deployment ---
box(78, 17, 21, 13, "Edge deployment:\nunder 0.002 ms/sample,\n$<$0.3M parameters\nsubstation gateway",
    fc="#fff2e6", ec=ORANGE, fs=7, tc="black")
arrow(73, 21.5, 78, 21.5, color=ORANGE, lw=1.2, ls="--")
ax.text(75.5, 25.5, "deploy", fontsize=6.4, color=ORANGE, ha="center")

fig.tight_layout(pad=0.3)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "fig_framework.png")
fig.savefig(path, bbox_inches="tight")
print("wrote", path)
