"""Step 7: ROC and precision-recall at realistic (rare) attack rates, for the k-mer model (k=6) and our 'wide' CNN.

WHO DID WHAT
  Authors: report only accuracy / confusion matrix on a 50% clean, 50% injected test set.
  Us: ROC curve (true-positive rate vs false-positive rate; does NOT depend on how rare attacks are) and precision at prevalence pi
    computed from it:  precision = TPR*pi / (TPR*pi + FPR*(1-pi)).  We do not resample the test set; this is exact for any pi
    if the clean/injected test reads are representative. Test = hosts never seen in training (host split, seed 0).
  Limits: test set has ~1000 clean reads (250 hosts x 4 datasets that reuse the same hosts), so FPR below ~0.001 is not measurable;
    when FPR = 0 we use 0.5/N_clean as a placeholder (marked *).
Usage (venv python): python -I ap_step7_roc_pr.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MaxAbsScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ap_step5_kmer_check as s5
import ap_step6_cnn as c

SETTINGS = [(5, 5), (1, 5)]  # (fragment size, retention); MaxNW, key 0
PREV = [0.5, 1e-2, 1e-3, 1e-5]


def kmer_scores(size, ret, k=6):
    reads, y, hosts = s5.load(size, ret, 0, "best")
    X, _ = s5.kmer_features(reads, k)
    test = s5.host_split(hosts, 0)
    sc = MaxAbsScaler().fit(X[~test])
    clf = LogisticRegression(max_iter=300).fit(sc.transform(X[~test]), y[~test])
    return y[test], clf.decision_function(sc.transform(X[test]))


def cnn_scores(size, ret):
    _, y, p = c.run(size, ret, "best", "wide", "host", 150, seed=0, verbose=False, return_scores=True)
    return y, p


def precision_at(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


def table(name, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    n_neg = int((y == 0).sum())
    rows = []
    for target in (0.9, 0.5):
        i = np.argmax(tpr >= target)
        f, star = (fpr[i], "") if fpr[i] > 0 else (0.5 / n_neg, "*")
        rows.append((target, f, star, [precision_at(tpr[i], f, pi) for pi in PREV]))
    print(f"  {name}")
    print("     at TPR  FPR       " + "  ".join(f"prec@pi={pi:g}".rjust(14) for pi in PREV))
    for target, f, star, pr in rows:
        print(f"     {target:.0%}   {f:8.4f}{star:1s}  " + "  ".join(f"{100 * v:13.4f}%" for v in pr))
    return fpr, tpr


def main():
    fig, axes = plt.subplots(len(SETTINGS), 2, figsize=(11, 4.6 * len(SETTINGS)))
    colors = {"k-mer (k=6)": "#C44E52", "wide CNN": "#4C72B0"}
    for r, (size, ret) in enumerate(SETTINGS):
        print(f"\nfragment size {size}, retention {ret}, MaxNW")
        for name, fn in (("k-mer (k=6)", kmer_scores), ("wide CNN", cnn_scores)):
            y, s = fn(size, ret)
            fpr, tpr = table(name, y, s)
            axes[r, 0].plot(np.maximum(fpr, 1e-4), tpr, color=colors[name], lw=2, label=name)
            for pi, ls in ((1e-2, "-"), (1e-3, "--"), (1e-5, ":")):
                m = tpr > 0
                axes[r, 1].plot(tpr[m], precision_at(tpr[m], np.maximum(fpr[m], 0.5 / (y == 0).sum()), pi), ls, color=colors[name], lw=2,
                                label=f"{name}, pi={pi:g}")
        ax = axes[r, 0]
        ax.plot([1e-4, 1], [1e-4, 1], color="gray", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("false-positive rate (clean reads flagged)"); ax.set_ylabel("true-positive rate (Trojans caught)")
        ax.set_title(f"ROC: fragment {size}, retention {ret}"); ax.legend(loc="lower right")
        ax = axes[r, 1]
        ax.set_xlabel("recall (Trojans caught)"); ax.set_ylabel("precision (flagged reads that are real Trojans)")
        ax.set_yscale("log"); ax.set_title(f"Precision-recall at attack rate pi: fragment {size}, retention {ret}"); ax.legend(fontsize=7, loc="lower left")
    for ax in axes.ravel():
        ax.set_facecolor("#EAEAF2"); ax.grid(color="white", lw=1); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    out = HERE.parent / "figures"
    for ext in ("png", "svg"):
        fig.savefig(out / f"step7_roc_pr.{ext}", dpi=150)
    print("\nsaved figures/step7_roc_pr.png and .svg")


if __name__ == "__main__":
    main()
