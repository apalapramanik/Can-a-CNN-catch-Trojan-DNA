"""Step 5 figure: how well does a plain k-mer model detect injected reads? (our own figure, drawn in the style of the paper's plots)

Panels: (a) accuracy vs fragment size for k = 3..6 (retention 0, MaxNW);  (b) accuracy vs retention for k = 6, fragment size 1 and 5,
with the paper's CNN points read off Fig. 8a (approximate, by eye);  (c) 2-D PCA of 6-mer frequencies, clean vs injected (size 5, MaxNW).
Test hosts are never seen in training. Saved as PNG and SVG in figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step5_kmer_check as s

COL = ["#5f9ea0", "#8b0000", "#daa520", "#6a5acd", "#2e8b57"]
BG = "#e5e5e5"
res = json.loads((ROOT / "results/step5_kmer.json").read_text())  # saved by ap_step5_kmer_check.py: key "k|size|scenario" -> [acc, auc]


def style(ax, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color="white", lw=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# retention sweep for panel (b) (k = 6, MaxNW, no encryption)
ret = {size: [100 * s.evaluate(size, 6, "best", retention=r)[0] for r in range(6)] for size in (1, 5)}
(ROOT / "results/step5_retention.json").write_text(json.dumps(ret))
paper_cnn = {1: [None, 100, 100, 100, 100, 100], 5: [None, 99, 97, 70, 52, 56]}  # Fig. 8a read by eye, retention 0 not shown there

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
# (a)
ax = axs[0]
w = 0.2
for i, k in enumerate((3, 4, 5, 6)):
    ax.bar(np.arange(5) + (i - 1.5) * w, [100 * res[f"{k}|{size}|MaxNW"][0] for size in range(1, 6)], w, color=COL[i], label=f"k = {k}")
ax.axhline(50, color="black", lw=0.8, ls="--", label="chance")
ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_ylim(40, 101)
ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, columnspacing=1)
style(ax, "(a) Fragment size (retention 0)", "Test accuracy (%)")
# (b)
ax = axs[1]
for size, c in ((1, COL[0]), (5, COL[4])):
    ax.plot(range(6), ret[size], "o-", color=c, label=f"k-mer model, fragment size {size}")
    pts = [(r, v) for r, v in enumerate(paper_cnn[size]) if v is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "D", mfc="none", color=c, ms=8, label=f"paper CNN, size {size} (approx.)")
ax.axhline(50, color="black", lw=0.8, ls="--")
ax.set_ylim(40, 105); ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
style(ax, "(b) Retention position (k = 6)", "Test accuracy (%)")
# (c)
ax = axs[2]
reads, y, hosts = s.load(5, 0, 0, "best", n_datasets=1)
X, _ = s.kmer_features(reads, 6)
P = PCA(2, random_state=0).fit_transform(X.toarray())
for lab, c, name in ((0, "#444444", "clean"), (1, COL[1], "injected (MaxNW)")):
    ax.scatter(P[y == lab, 0], P[y == lab, 1], s=7, alpha=0.6, color=c, label=name, linewidths=0)
ax.legend(frameon=False, fontsize=8, markerscale=2)
style(ax, "(c) PCA 1 of 6-mer frequencies (size 5, retention 0)", "PCA 2")
ax.grid(axis="x", color="white", lw=1)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(ROOT / f"figures/step5_kmer_check.{ext}", dpi=160)
print("saved figures/step5_kmer_check.png/.svg")
print("retention sweep k=6:", {k: [round(x, 1) for x in v] for k, v in ret.items()})
