"""Step 17: redraw EVERY figure of the paper with very large fonts and save PNG + PDF (+ SVG) in figures/.
Nothing is retrained or re-simulated: bars come from results/fig5b.json, fig6.json, step5_*.json, step14_*.json, step8_report_numbers.json
and results/step9_authors_cnn; ROC/PR curves come from the saved step-8 weights in models/ (same host split, seed 0).
The originals (small fonts) are kept in figures/old_small/.
Usage (venv python): ~/venvs/ap_trojan/bin/python -I ap_step17_big_figs.py [name ...]   names: fig5b fig6 step5 step7 step8 step15 step14
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
FIG, RES = ROOT / "figures", ROOT / "results"

plt.rcParams.update({"font.size": 22, "axes.labelsize": 24, "axes.titlesize": 24, "xtick.labelsize": 21, "ytick.labelsize": 21,
                     "legend.fontsize": 19, "legend.title_fontsize": 20, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
BG = "#e5e5e5"
BG2 = "#EAEAF2"
PAL = ["#5f9ea0", "#8b0000", "#daa520", "#6a5acd", "#2e8b57"]
RED, BLUE = "#C44E52", "#4C72B0"


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def style(ax, bg=BG, xgrid=False):
    ax.set_facecolor(bg)
    ax.grid(axis="both" if xgrid else "y", color="white", lw=1.4)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


# ------------------------------------------------------------------ fig 5b and 6 (paper style bars)
def bars(ax, groups, series, get, labels, hatches=False, width=0.15):
    for k, s in enumerate(series):
        xs = np.arange(len(groups)) + (k - (len(series) - 1) / 2) * width
        vals = [np.mean(get(g, s)["mean"]) for g in groups]
        errs = [np.mean(get(g, s)["sd"]) for g in groups]
        ax.bar(xs, vals, width, yerr=errs, color=s[0] if hatches else PAL[k % 5], label=labels[k], hatch=s[1] if hatches else None,
               edgecolor="#444" if hatches else "white", linewidth=1.0 if hatches else 0, error_kw=dict(ecolor="black", lw=2, capsize=4))
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)


def fig5b():
    raw = json.loads((RES / "fig5b.json").read_text())
    d = {n: [{"mean": raw["mean"][n][i], "sd": raw["sd"][n][i]} for i in range(5)] for n in raw["mean"]}
    for name, groups in (("fig5b_reproduction", ["MaxNW (paper metric)", "Random", "MinNW"]),
                         ("fig5b_with_full_maxnw", ["MaxNW (paper metric)", "MaxNW (full payload)", "Random", "MinNW"])):
        fig, ax = plt.subplots(figsize=(13 if len(groups) == 4 else 10.5, 6.6))
        bars(ax, groups, range(5), lambda g, s: d[g][s], [str(i) for i in range(1, 6)], width=0.15)
        style(ax)
        ax.set_ylabel("NW Score Reduction (%)"); ax.set_xlabel("(b) Insertion Technique")
        ax.set_xticklabels([g.replace(" (", "\n(") for g in groups])
        ax.legend(title="Fragment Size", ncol=5, frameon=False, loc="upper left")
        ax.set_ylim(0, 27)
        save(fig, name)


def fig6():
    d = json.loads((RES / "fig6.json").read_text())
    fig, axs = plt.subplots(2, 1, figsize=(12, 12.5))
    bars(axs[0], [str(s) for s in range(1, 6)], range(1, 6), lambda g, r: d["a"][str(r)][int(g) - 1], [str(r) for r in range(1, 6)])
    style(axs[0]); axs[0].set_ylabel("NW Score Reduction (%)"); axs[0].set_xlabel("(a) Fragment Size")
    axs[0].legend(title="Retention Position (key=50)", ncol=5, frameon=False, loc="upper left", columnspacing=1.0, handletextpad=0.4)
    axs[0].set_ylim(0, 5.2)
    keys = [0, 10, 20, 30, 40, 50]
    combos = [(PAL[0], ".."), (PAL[0], "//"), (PAL[4], ".."), (PAL[4], "//")]
    keyed = [(1, 1), (1, 5), (5, 1), (5, 5)]
    bars(axs[1], [("No Encryption" if k == 0 else f"key={k}") for k in keys], combos,
         lambda g, s: d["b"][f"{0 if g == 'No Encryption' else int(g[4:])}|{keyed[combos.index(s)][0]}|{keyed[combos.index(s)][1]}"],
         [f"size {a}, ret. {b}" for a, b in keyed], hatches=True, width=0.18)
    style(axs[1]); axs[1].set_ylabel("NW Score Reduction (%)"); axs[1].set_xlabel("(b) Encryption Key")
    axs[1].set_xticklabels(["None" if k == 0 else f"{k}" for k in keys])
    axs[1].legend(ncol=2, frameon=False, loc="upper left")
    axs[1].set_ylim(0, 5.6)
    fig.tight_layout()
    save(fig, "fig6_reproduction")


# ------------------------------------------------------------------ step 5 (k-mer)
def step5():
    import ap_step5_kmer_check as s
    res = json.loads((RES / "step5_kmer.json").read_text())
    ret = json.loads((RES / "step5_retention.json").read_text())
    paper_cnn = {1: [None, 100, 100, 100, 100, 100], 5: [None, 99, 97, 70, 52, 56]}
    fig, axs = plt.subplots(3, 1, figsize=(12, 22))
    ax = axs[0]; w = 0.2
    for i, k in enumerate((3, 4, 5, 6)):
        ax.bar(np.arange(5) + (i - 1.5) * w, [100 * res[f"{k}|{size}|MaxNW"][0] for size in range(1, 6)], w, color=PAL[i], label=f"k = {k}")
    ax.axhline(50, color="black", lw=1.5, ls="--", label="chance")
    ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_ylim(40, 112)
    ax.legend(frameon=False, ncol=5, loc="upper center", columnspacing=1.0, handletextpad=0.4, fontsize=17)
    style(ax); ax.set_xlabel("(a) Fragment Size (Retention 0)"); ax.set_ylabel("Test Accuracy (%)")
    ax = axs[1]
    for size, c in ((1, PAL[0]), (5, PAL[4])):
        ax.plot(range(6), ret[str(size)], "o-", color=c, lw=3.5, ms=11, label=f"k-mer, size {size}")
        pts = [(r, v) for r, v in enumerate(paper_cnn[size]) if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "D", mfc="none", mew=3, color=c, ms=14, label=f"paper CNN, size {size} (approx.)")
    ax.axhline(50, color="black", lw=1.5, ls="--")
    ax.set_ylim(40, 125); ax.set_yticks([40, 60, 80, 100])
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=17, columnspacing=1.0)
    style(ax); ax.set_xlabel("(b) Retention Position (k = 6)"); ax.set_ylabel("Test Accuracy (%)")
    ax = axs[2]
    reads, y, hosts = s.load(5, 0, 0, "best", n_datasets=1)
    X, _ = s.kmer_features(reads, 6)
    P = PCA(2, random_state=0).fit_transform(X.toarray())
    for lab, c, name in ((0, "#444444", "clean"), (1, PAL[1], "injected (MaxNW)")):
        ax.scatter(P[y == lab, 0], P[y == lab, 1], s=26, alpha=0.6, color=c, label=name, linewidths=0)
    ax.legend(frameon=False, markerscale=2.5, loc="upper right")
    style(ax, xgrid=True); ax.set_xlabel("(c) PCA 1 of 6-mer Frequencies"); ax.set_ylabel("PCA 2")
    ax.set_xticks([-0.01, 0, 0.01, 0.02]); ax.set_xticklabels(["-0.01", "0", "0.01", "0.02"])
    fig.tight_layout()
    save(fig, "step5_kmer_check")


# ------------------------------------------------------------------ CNN scores (saved weights, no retraining)
def cnn_scores(arch, size, ret, seed=0):
    import ap_step5_kmer_check as s5
    import ap_step6_cnn as c
    reads, y, hosts = s5.load(size, ret, 0, "best", 4)
    test = s5.host_split(hosts, seed)
    X = torch.from_numpy(c.one_hot(reads))[torch.from_numpy(test)]
    model = c.build_model(arch)
    model.load_state_dict(torch.load(ROOT / f"models/{arch}_size{size}_ret{ret}_best_key0_host_seed{seed}_ep150.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(X[i:i + 500]).squeeze(1) for i in range(0, len(X), 500)])
    return y[test], p.numpy()


def prec(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


def authors_acc():
    A = {}
    for f in sorted((RES / "step9_authors_cnn").glob("size*_ds0_ep3000.txt")):
        sz, rt = f.name.split("_")[0][4:], f.name.split("_")[1][3:]
        A.setdefault(int(sz), {})[int(rt)] = float(f.read_text().split("\n")[1].split(",")[-1])
    return A


def accuracy_fig(archs, name):
    num = json.loads((RES / "step8_report_numbers.json").read_text())["accuracy_auc"]
    A = authors_acc()
    lab = {"paper": "Reimplemented CNN", "wide": "Wide CNN"}
    col = {"paper": RED, "wide": BLUE}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 13), sharex=True)
    for ax, size in zip(axes, (1, 5)):
        for a in archs:
            ax.plot(range(6), [100 * num[f"size{size}_ret{r}_{a}"][0] for r in range(6)], "o-", color=col[a], lw=3.5, ms=11, label=lab[a])
        ax.plot(sorted(A[size]), [A[size][r] for r in sorted(A[size])], "s--", color="black", lw=3, ms=11, label="Original CNN")
        ax.axhline(50, color="gray", lw=2, ls=":"); ax.set_ylim(20, 101)
        ax.set_title(f"Fragment Size {size}"); ax.set_ylabel("Test Accuracy (%)"); style(ax, BG2, xgrid=True)
    axes[1].set_xlabel("Retention")
    axes[0].legend(loc="lower left", fontsize=17)
    fig.tight_layout()
    save(fig, name)


def roc_pr_fig(archs, name):
    col = {"paper": RED, "wide": BLUE}
    nm = {"paper": "Reimplemented CNN", "wide": "Wide CNN"}
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for r, (size, ret) in enumerate([(1, 5), (5, 5)]):
        for a in archs:
            y, p = cnn_scores(a, size, ret)
            fpr, tpr, _ = roc_curve(y, p)
            n_neg = int((y == 0).sum())
            axes[r, 0].plot(np.maximum(fpr, 1e-3), tpr, color=col[a], lw=3.5, label=nm[a])
            m = tpr > 0
            f = np.maximum(fpr[m], 0.5 / n_neg)
            axes[r, 1].plot(tpr[m], prec(tpr[m], f, 0.5), ":", color=col[a], lw=3.5, label=f"{nm[a]}: 50/50")
            axes[r, 1].plot(tpr[m], prec(tpr[m], f, 1e-3), "-", color=col[a], lw=3.5, label=f"{nm[a]}: 1 in 1,000")
        axes[r, 0].plot([1e-3, 1], [1e-3, 1], color="gray", lw=1.8)
        axes[r, 0].set_xscale("log"); axes[r, 0].set_xlabel("False-Positive Rate"); axes[r, 0].set_ylabel("True-Positive Rate")
        axes[r, 0].set_title(f"ROC, Fragment {size}, Retention {ret}")
        axes[r, 1].set_yscale("log"); axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
        axes[r, 1].set_title(f"Precision-Recall, Fragment {size}, Retention {ret}")
        axes[r, 1].set_ylim(5e-4, 8)
        style(axes[r, 0], BG2, True); style(axes[r, 1], BG2, True)
    fig.tight_layout()
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=20)
    save(fig, name)


def step8():
    accuracy_fig(("paper", "wide"), "step8_accuracy")
    roc_pr_fig(("paper", "wide"), "step8_roc_pr")


def accuracy_merged(archs, name):
    num = json.loads((RES / "step8_report_numbers.json").read_text())["accuracy_auc"]
    A = authors_acc()
    lab = {"paper": "Reimplemented CNN", "wide": "Wide CNN"}
    col = {"paper": RED, "wide": BLUE}
    ls = {1: "-", 5: "--"}
    mk = {1: "o", 5: "^"}
    fig, ax = plt.subplots(figsize=(12, 8.5))
    for size in (1, 5):
        for a in archs:
            ax.plot(range(6), [100 * num[f"size{size}_ret{r}_{a}"][0] for r in range(6)], ls[size], marker=mk[size], color=col[a],
                    lw=3.5, ms=11, label=f"{lab[a]}, Fragment {size}")
        ax.plot(sorted(A[size]), [A[size][r] for r in sorted(A[size])], ls[size], marker="s", color="black", lw=3, ms=10,
                label=f"Original CNN, Fragment {size}")
    ax.axhline(50, color="gray", lw=2, ls=":", label="Chance (50%)")
    ax.set_ylim(20, 101); ax.set_xlabel("Retention"); ax.set_ylabel("Test Accuracy (%)")
    style(ax, BG2, xgrid=True)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=19)
    fig.tight_layout()
    save(fig, name)


def step15():
    accuracy_merged(("paper",), "step15_paper_accuracy")
    roc_pr_fig(("paper",), "step15_paper_roc_pr")


def step7():
    import ap_step7_roc_pr as m7
    col = {"k-mer (k=6)": RED, "wide CNN": BLUE}
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for r, (size, ret) in enumerate([(5, 5), (1, 5)]):
        for nm in ("k-mer (k=6)", "wide CNN"):
            y, s = m7.kmer_scores(size, ret) if nm.startswith("k-mer") else cnn_scores("wide", size, ret)
            fpr, tpr, _ = roc_curve(y, s)
            axes[r, 0].plot(np.maximum(fpr, 1e-4), tpr, color=col[nm], lw=3.5, label=nm)
            for pi, ls in ((1e-2, "-"), (1e-3, "--"), (1e-5, ":")):
                m = tpr > 0
                axes[r, 1].plot(tpr[m], prec(tpr[m], np.maximum(fpr[m], 0.5 / (y == 0).sum()), pi), ls, color=col[nm], lw=3,
                                label=f"{nm.split()[0]}, " + r"$\pi$=" + f"{pi:g}")
        axes[r, 0].plot([1e-4, 1], [1e-4, 1], color="gray", lw=1.8)
        axes[r, 0].set_xscale("log"); axes[r, 0].set_xlabel("False-Positive Rate"); axes[r, 0].set_ylabel("True-Positive Rate")
        axes[r, 0].set_title(f"ROC, Fragment {size}, Retention {ret}")
        axes[r, 1].set_yscale("log"); axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
        axes[r, 1].set_title(f"Precision-Recall, Fragment {size}, Retention {ret}")
        style(axes[r, 0], BG2, True); style(axes[r, 1], BG2, True)
    fig.tight_layout()
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=20)
    save(fig, "step7_roc_pr")


# ------------------------------------------------------------------ step 14 (amplification lines)
def step14():
    d = json.loads((RES / "step14_full_dataset_boxplot.json").read_text())
    Ns, M = d["Ns"], d["machines"]
    cols = ["#C44E52", "#DD8452", "#E8A33D", "#55A868", "#4C72B0", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(15, 9))
    for c, (name, pe) in zip(cols, M.items()):
        bm = np.array([d["malware_blocked_mean"][name][str(N)] for N in Ns]); bs = np.array([d["malware_blocked_std"][name][str(N)] for N in Ns])
        lm = np.array([d["legit_quality_mean"][name][str(N)] for N in Ns])
        ax.plot(Ns, bm, "o-", color=c, lw=4, ms=11, label=f"{name.replace(' (', ' ').replace(')', '')}, {100 * pe:.2f}%")
        ax.fill_between(Ns, bm - bs, bm + bs, color=c, alpha=0.12)
        ax.plot(Ns, lm, "o:", color=c, lw=4, ms=11, alpha=0.65)
    ax.plot([], [], "k-", lw=4, label="Solid: Malware Blocked"); ax.plot([], [], "k:", lw=4, label="Dotted: Legitimate Decode")
    ax.set_xlabel("Number Of Copies"); ax.set_ylabel("Percentage (%)"); ax.set_xticks(Ns)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=19, title="Platform and Error Rate")
    style(ax, BG2, True)
    save(fig, "step14_full_dataset_lines")


ALL = {"fig5b": fig5b, "fig6": fig6, "step5": step5, "step7": step7, "step8": step8, "step15": step15, "step14": step14}
if __name__ == "__main__":
    for n in (sys.argv[1:] or ALL):
        ALL[n]()
