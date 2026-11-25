"""Step 19: grouped bar chart of test accuracy for the Reimplemented CNN (Fig. 7b model) and our Wide CNN.

Data: results/step18_same_as_original/*  (pooled 10 datasets, read level split, SGD, early stopping, one seed).
Bars: fragment size 1 and 5, retention 1 and 5. Writes figures/step19_accuracy_bars (both models) and
figures/step19_accuracy_bars_reimplemented (Reimplemented CNN only), each as png, pdf and svg.
Usage (venv python): ~/venvs/ap_trojan/bin/python -I ap_step19_accuracy_bars.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ap_step17_big_figs as big  # fonts, colours, style() and save()

RES = HERE.parent / "results/step18_same_as_original"
CELLS = [(1, 1), (1, 5), (5, 1), (5, 5)]  # (fragment size, retention)


def acc(size, ret, wide):
    f = RES / f"size{size}_ret{ret}_line_pool10_es100m500{'_wide' if wide else ''}_ep1000_seed0.txt"
    return float(f.read_text().split("\n")[1].split(",")[-1]) if f.exists() else None


def bars(series, name):
    fig, ax = plt.subplots(figsize=(13, 8))
    w = 0.8 / len(series)
    for k, (label, color, wide) in enumerate(series):
        xs = np.arange(len(CELLS)) + (k - (len(series) - 1) / 2) * w
        vals = [acc(s, r, wide) for s, r in CELLS]
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax.bar(x, v, w * 0.92, color=color, label=label if x == xs[[i for i, y in enumerate(vals) if y is not None][0]] else None)
            ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=19)
    ax.axhline(50, color="gray", lw=2, ls=":", label="Chance (50%)")
    ax.set_xticks(np.arange(len(CELLS)))
    ax.set_xticklabels([f"Fragment {s}\nRetention {r}" for s, r in CELLS])
    ax.set_ylim(0, 112); ax.set_ylabel("Test Accuracy (%)")
    big.style(ax, big.BG2, xgrid=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=20)
    fig.tight_layout()
    big.save(fig, name)


if __name__ == "__main__":
    bars([("Reimplemented CNN", big.RED, False)], "step19_accuracy_bars_reimplemented")
    bars([("Reimplemented CNN", big.RED, False), ("Wide CNN", big.BLUE, True)], "step19_accuracy_bars")
