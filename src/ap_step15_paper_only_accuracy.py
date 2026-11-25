"""Step 15b: accuracy vs retention for ONLY the authors' CNN code (run by us) and the paper-style CNN (our training); numbers come from
results/step8_report_numbers.json and results/step9_authors_cnn (no retraining). The wide CNN is left out (it appears in the future-directions comparison).
Usage: python3 -I ap_step15_paper_only_accuracy.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
num = json.loads((ROOT / "results/step8_report_numbers.json").read_text())["accuracy_auc"]
AUTH = {}
for f in sorted((ROOT / "results/step9_authors_cnn").glob("size*_ds0_ep3000.txt")):
    sz, rt = f.name.split("_")[0][4:], f.name.split("_")[1][3:]
    AUTH.setdefault(int(sz), {})[int(rt)] = float(f.read_text().split("\n")[1].split(",")[-1])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, size in zip(axes, (1, 5)):
    ax.plot(range(6), [100 * num[f"size{size}_ret{r}_paper"][0] for r in range(6)], "o-", color="#C44E52", lw=2, label="Paper-Style CNN (Our Training)")
    ax.plot(sorted(AUTH[size]), [AUTH[size][r] for r in sorted(AUTH[size])], "s--", color="black", label="Authors' CNN Code, Run By Us (Line Split)")
    ax.axhline(50, color="gray", lw=1, ls=":"); ax.set_ylim(20, 101)
    ax.set_xlabel("Retention"); ax.set_title(f"Fragment Size {size}")
    ax.set_facecolor("#EAEAF2"); ax.grid(color="white", lw=1); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
axes[0].set_ylabel("Test Accuracy (%)"); axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
for e in ("png", "svg"): fig.savefig(ROOT / f"figures/step15_paper_accuracy.{e}", dpi=150)
print("saved")
