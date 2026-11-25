"""Step 8: figures + numbers for the report: paper-style CNN vs our best ('wide') CNN.
Both: host split, MaxNW, key 0, 4 datasets pooled, Adam, 150 epochs, seed 0 (one seed). 'paper' = paper-text architecture (our training).
Paper's own accuracies (Fig. 8a, read off by eye, size 5 only) are overlaid as markers.
Usage (venv python): python -I ap_step8_report_figs.py   (several minutes)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ap_step6_cnn as c

ROOT = HERE.parent
SIZES, RETS, ARCHS = (1, 5), range(6), ("paper", "wide")
# exact test accuracies from RUNNING the authors' trojan_cnn.py (ap_step9_authors_cnn.py; their line split, dataset 0, 3000 epochs)
AUTH = {}
for f in sorted((ROOT / "results/step9_authors_cnn").glob("size*_ds0_ep3000.txt")):
    sz, rt = f.name.split("_")[0][4:], f.name.split("_")[1][3:]
    AUTH.setdefault(int(sz), {})[int(rt)] = float(f.read_text().split("\n")[1].split(",")[-1])
COL = {"paper": "#C44E52", "wide": "#4C72B0"}
LAB = {"paper": "Paper-Style CNN (Our Training)", "wide": "Our Best CNN (Wide, Global Max-Pool)"}
FOCUS = [(1, 5), (5, 5)]
NAME = {"paper": "Paper-Style", "wide": "Our Best"}


def style(ax):
    ax.set_facecolor("#EAEAF2"); ax.grid(color="white", lw=1); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)


def prec(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


res, scores = {}, {}
for size in SIZES:
    for ret in RETS:
        for arch in ARCHS:
            h, y, p = c.run(size, ret, "best", arch, "host", 150, seed=0, verbose=False, return_scores=True, save_dir=ROOT / "models")
            res[(size, ret, arch)] = h[-1]
            if (size, ret) in FOCUS: scores[(size, ret, arch)] = (y, p)
            print(size, ret, arch, f"acc {h[-1][0]*100:.1f} auc {h[-1][1]:.3f}", flush=True)

# ---- figure 1: accuracy
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, size in zip(axes, SIZES):
    for arch in ARCHS:
        ax.plot(list(RETS), [100 * res[(size, r, arch)][0] for r in RETS], "o-", color=COL[arch], lw=2, label=LAB[arch])
    if size in AUTH:
        ax.plot(sorted(AUTH[size]), [AUTH[size][r] for r in sorted(AUTH[size])], "s--", color="black", label="Authors' CNN Code, Run By Us (Line Split)")
    ax.axhline(50, color="gray", lw=1, ls=":"); ax.set_ylim(20, 101)
    ax.set_xlabel("Retention"); ax.set_title(f"Fragment Size {size}"); style(ax)
axes[0].set_ylabel("Test Accuracy (%)"); axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
for e in ("png", "svg"): fig.savefig(ROOT / f"figures/step8_accuracy.{e}", dpi=150)

# ---- figure 2: ROC and PR
fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
for r, (size, ret) in enumerate(FOCUS):
    for arch in ARCHS:
        y, p = scores[(size, ret, arch)]
        fpr, tpr, _ = roc_curve(y, p)
        n_neg = int((y == 0).sum())
        axes[r, 0].plot(np.maximum(fpr, 1e-3), tpr, color=COL[arch], lw=2, label=LAB[arch])
        m = tpr > 0
        f = np.maximum(fpr[m], 0.5 / n_neg)
        axes[r, 1].plot(tpr[m], prec(tpr[m], f, 0.5), ":", color=COL[arch], lw=2, label=f"{NAME[arch]}: Balanced 50/50")
        axes[r, 1].plot(tpr[m], prec(tpr[m], f, 1e-3), "-", color=COL[arch], lw=2, label=f"{NAME[arch]}: 1 Trojan per 1,000")
    axes[r, 0].plot([1e-3, 1], [1e-3, 1], color="gray", lw=1)
    axes[r, 0].set_xscale("log"); axes[r, 0].set_xlabel("False-Positive Rate"); axes[r, 0].set_ylabel("True-Positive Rate")
    axes[r, 0].set_title(f"ROC, Fragment {size}, Retention {ret}")
    if r == 0: axes[r, 0].legend(fontsize=7, loc="lower right")
    axes[r, 1].set_yscale("log"); axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
    axes[r, 1].set_title(f"Precision-Recall, Fragment {size}, Retention {ret}")
    axes[r, 1].legend(fontsize=7, loc="lower left")
    style(axes[r, 0]); style(axes[r, 1])
fig.tight_layout()
for e in ("png", "svg"): fig.savefig(ROOT / f"figures/step8_roc_pr.{e}", dpi=150)

# ---- numbers
out = {"accuracy_auc": {f"size{s}_ret{r}_{a}": list(map(float, res[(s, r, a)])) for (s, r, a) in res}, "operating_points": {}}
for (size, ret, arch), (y, p) in scores.items():
    fpr, tpr, _ = roc_curve(y, p)
    n_neg = int((y == 0).sum())
    i = int(np.argmax(tpr >= 0.9))
    f = fpr[i] if fpr[i] > 0 else 0.5 / n_neg
    out["operating_points"][f"size{size}_ret{ret}_{arch}"] = {
        "tpr": float(tpr[i]), "fpr": float(f), "fpr_was_zero": bool(fpr[i] == 0), "n_clean_test": n_neg, "n_trojan_test": int((y == 1).sum()),
        "precision": {str(pi): float(prec(tpr[i], f, pi)) for pi in (0.5, 1e-2, 1e-3, 1e-5)}}
(ROOT / "results/step8_report_numbers.json").write_text(json.dumps(out, indent=1))
print("done")
