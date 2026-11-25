"""Step 20: ROC and precision recall curves for the Reimplemented CNN (Fig. 7b model) and our Wide CNN.

Data: test scores saved by ap_step18 (results/step18_same_as_original/*.npz): 2,500 clean + 2,500 Trojan test reads,
pooled 10 datasets, read level split, one seed. Precision at attack rate pi is TPR*pi / (TPR*pi + FPR*(1-pi)).
A false positive rate of 0 is plotted at 0.5/2,500 because the test set has only 2,500 clean reads.
Writes figures/step20_roc_pr_ret{r} (both models) and figures/step20_roc_pr_ret{r}_reimplemented, each as png, pdf, svg.
Usage (venv python): ~/venvs/ap_trojan/bin/python -I ap_step20_roc_pr.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ap_step17_big_figs as big

RES = HERE.parent / "results/step18_same_as_original"


def load(size, ret, wide):
    f = RES / f"size{size}_ret{ret}_line_pool10_es100m500{'_wide' if wide else ''}_ep1000_seed0.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return d["y"], d["score"]


def prec(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


def fig(ret, models, name):
    f, axes = plt.subplots(2, 2, figsize=(17, 13))
    for r, size in enumerate((1, 5)):
        for label, color, wide in models:
            d = load(size, ret, wide)
            if d is None:
                continue
            y, s = d
            fpr, tpr, _ = roc_curve(y, s)
            floor = 0.5 / int((y == 0).sum())
            ff = np.maximum(fpr, floor)
            axes[r, 0].plot(ff, tpr, color=color, lw=3.5, label=label)
            m = tpr >= 0.05  # the first points sit on the false positive floor and make the curve start with a ramp
            axes[r, 1].plot(tpr[m], prec(tpr[m], ff[m], 0.5), ":", color=color, lw=3.5, label=f"{label}: 50/50")
            axes[r, 1].plot(tpr[m], prec(tpr[m], ff[m], 1e-3), "-", color=color, lw=3.5, label=f"{label}: 1 in 1,000")
        axes[r, 0].plot([floor, 1], [floor, 1], color="gray", lw=1.8)
        axes[r, 0].set_xscale("log"); axes[r, 0].set_xlabel("False-Positive Rate"); axes[r, 0].set_ylabel("True-Positive Rate")
        axes[r, 0].set_title(f"ROC, Fragment {size}, Retention {ret}")
        axes[r, 1].set_yscale("log"); axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
        axes[r, 1].set_title(f"Precision-Recall, Fragment {size}, Retention {ret}")
        axes[r, 1].set_ylim(5e-4, 8)
        big.style(axes[r, 0], big.BG2, True); big.style(axes[r, 1], big.BG2, True)
    f.tight_layout()
    h, l = axes[0, 1].get_legend_handles_labels()
    f.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=20)
    big.save(f, name)


if __name__ == "__main__":
    R, W = ("Reimplemented CNN", big.RED, False), ("Wide CNN", big.BLUE, True)
    for ret in (5, 1):
        fig(ret, [R], f"step20_roc_pr_ret{ret}_reimplemented")
        fig(ret, [R, W], f"step20_roc_pr_ret{ret}")
