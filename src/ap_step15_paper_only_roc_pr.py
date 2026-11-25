"""Step 15: ROC + precision-recall for ONLY the paper-style CNN (loads the saved weights from step 8; no retraining).
Same data, split (host), seed and plot style as ap_step8_report_figs.py; the wide CNN is left out so the analysis section can show the paper's detector alone.
Usage (venv python): python -I ap_step15_paper_only_roc_pr.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ap_step5_kmer_check as s5
import ap_step6_cnn as c

ROOT = HERE.parent
FOCUS = [(1, 5), (5, 5)]
RED = "#C44E52"


def style(ax):
    ax.set_facecolor("#EAEAF2"); ax.grid(color="white", lw=1); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)


def prec(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


def scores(size, ret, seed=0):
    reads, y, hosts = s5.load(size, ret, 0, "best", 4)
    test = s5.host_split(hosts, seed)
    X = torch.from_numpy(c.one_hot(reads))[torch.from_numpy(test)]
    model = c.build_model("paper")
    model.load_state_dict(torch.load(ROOT / f"models/paper_size{size}_ret{ret}_best_key0_host_seed{seed}_ep150.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(X[i:i + 500]).squeeze(1) for i in range(0, len(X), 500)])
    return y[test], p.numpy()


fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
for r, (size, ret) in enumerate(FOCUS):
    y, p = scores(size, ret)
    acc = ((p > 0) == (y == 1)).mean()
    print(size, ret, f"acc {acc*100:.1f}% (step 8 reported the same run)")
    fpr, tpr, _ = roc_curve(y, p)
    n_neg = int((y == 0).sum())
    axes[r, 0].plot(np.maximum(fpr, 1e-3), tpr, color=RED, lw=2, label="Paper-Style CNN (Our Training)")
    m = tpr > 0
    f = np.maximum(fpr[m], 0.5 / n_neg)
    axes[r, 1].plot(tpr[m], prec(tpr[m], f, 0.5), ":", color=RED, lw=2, label="Balanced 50/50")
    axes[r, 1].plot(tpr[m], prec(tpr[m], f, 1e-3), "-", color=RED, lw=2, label="1 Trojan per 1,000")
    axes[r, 0].plot([1e-3, 1], [1e-3, 1], color="gray", lw=1)
    axes[r, 0].set_xscale("log"); axes[r, 0].set_xlabel("False-Positive Rate"); axes[r, 0].set_ylabel("True-Positive Rate")
    axes[r, 0].set_title(f"ROC, Fragment {size}, Retention {ret}")
    axes[r, 1].set_yscale("log"); axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
    axes[r, 1].set_title(f"Precision-Recall, Fragment {size}, Retention {ret}")
    axes[r, 1].legend(fontsize=7, loc="lower left")
    style(axes[r, 0]); style(axes[r, 1])
axes[0, 0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
for e in ("png", "svg"): fig.savefig(ROOT / f"figures/step15_paper_roc_pr.{e}", dpi=150)
print("saved")
