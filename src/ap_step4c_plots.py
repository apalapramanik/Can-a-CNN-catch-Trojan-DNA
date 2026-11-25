"""Step 4c: reproduce paper Fig. 5b, 6a and 6b (NW score reduction) at full scale, drawn in the paper's style.

WHO DID WHAT
  Authors: the experiment (Fig. 5b, 6a, 6b), the settings in generate_dataset_new.py (tags TCATG/GTTGG, 10 datasets of 1000 hosts,
    payloads from seed_data_generator.py rearranged by make_shorter_payloads), and the plot design.
  Us: this re-run with our own code, and the following analysis of what the paper's MaxNW bars measure:
    the authors' greedy function reports np.max(running scores) as its 'best score'. Every fragment can only lower the score, so that
    maximum is the score after the FIRST fragment only. The paper's MaxNW bars match that number ("MaxNW, paper metric"), not the score
    of the finished read ("MaxNW, full payload"). Random and MinNW report the final read in both.
  Matrix: PAM10 (what the paper text says; with the authors' real tags and payloads it reproduces their Random and MinNW bars).

Usage:  python3 ap_step4c_plots.py 5b     (about 3 min)     python3 ap_step4c_plots.py 6     (about 3 min)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step3_encode as enc
import ap_step4_score as sc
import ap_step4b_greedy as gr
import ap_step4c_fig5b_sweep as sw

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
M = sc.load_matrix(ROOT / "matrices/PAM10")
TABLE = gr.loss_table(M)
N_DATASETS, N_HOSTS = 10, 1000
KEY_DEFAULT = 50
enc.START_TAG, enc.END_TAG = "TCATG", "GTTGG"  # the authors' experiment tags


def setup(seed=0):
    """Authors' seed data: first 1000 reads as hosts, 1000 payloads like 'xy42ab' (random.seed(0) in their scripts)."""
    hosts = sw.load_reads()[:N_HOSTS]
    rng = np.random.default_rng(seed)
    payloads = [sw.authors_payload(rng) for _ in range(N_HOSTS)]
    idx = np.array([[gr.LETTERS.index(c) for c in h] for h in hosts])
    base = np.array([sc.ungapped_score(h, h, M) for h in hosts])
    return hosts, payloads, idx, base


def first_fragment_reduction(host_idx, base, frag):
    """Reduction (%) after placing only the first fragment at its best position (the 'paper metric' for MaxNW)."""
    return 100 * gr.fragment_losses(host_idx, frag, TABLE).min() / base


# ---------------------------------------------------------------- Fig. 5b
def run_5b():
    hosts, payloads, idx, base = setup()
    names = ["MaxNW (paper metric)", "MaxNW (full payload)", "Random", "MinNW"]
    out = {n: [[] for _ in range(5)] for n in names}  # out[name][size-1] = list of 10 dataset means
    sd = {n: [[] for _ in range(5)] for n in names}  # same shape: std over the 1000 reads of each dataset (the paper's error bars)
    rng = np.random.default_rng(1)
    for d in range(N_DATASETS):
        order = rng.permutation(N_HOSTS)  # each dataset reshuffles the hosts (authors: random.sample)
        for size in range(1, 6):
            acc = {n: [] for n in names}
            for k, h in enumerate(order):  # payload k is paired with the k-th shuffled host (authors: zip(hosts, payloads))
                host, b = hosts[h], base[h]
                frags = enc.make_fragments(payloads[k], size, KEY_DEFAULT, 0)
                acc["MaxNW (paper metric)"].append(first_fragment_reduction(idx[h], b, frags[0]))
                for n, seq in (("MaxNW (full payload)", gr.place_greedy(host, frags, M, True)[0]),
                               ("Random", gr.place_random(host, frags, rng)[0]),
                               ("MinNW", gr.place_greedy(host, frags, M, False)[0])):
                    acc[n].append(100 * (b - sc.ungapped_score(host, seq, M)) / b)
            for n in names:
                out[n][size - 1].append(float(np.mean(acc[n])))
                sd[n][size - 1].append(float(np.std(acc[n])))
        print(f"  5b: dataset {d + 1}/{N_DATASETS} done", flush=True)
    (RES / "fig5b.json").write_text(json.dumps({"mean": out, "sd": sd}))


# ---------------------------------------------------------------- Fig. 6
def run_6():
    hosts, payloads, idx, base = setup()
    rng = np.random.default_rng(2)
    orders = [rng.permutation(N_HOSTS) for _ in range(N_DATASETS)]

    def cell(size, key, ret):  # (10 dataset means, 10 within-dataset stds over reads) of the paper-metric MaxNW reduction
        means, sds = [], []
        for order in orders:
            v = [first_fragment_reduction(idx[h], base[h], enc.make_fragments(payloads[k], size, key, ret)[0])
                 for k, h in enumerate(order)]
            means.append(float(np.mean(v)))
            sds.append(float(np.std(v)))
        return {"mean": means, "sd": sds}

    a = {str(r): [cell(s, KEY_DEFAULT, r) for s in range(1, 6)] for r in range(1, 6)}  # 6a: retention x size
    print("  6a done", flush=True)
    b = {}  # 6b: key x (size, retention)
    for key in (0, 10, 20, 30, 40, 50):
        for size in (1, 5):
            for ret in (1, 5):
                b[f"{key}|{size}|{ret}"] = cell(size, key, ret)
    print("  6b done", flush=True)
    (RES / "fig6.json").write_text(json.dumps({"a": a, "b": b}))


# ---------------------------------------------------------------- drawing (paper style)
def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, png_name):
        """Write the figure twice: PNG (quick look) and SVG (vector, scales without blur, editable in Illustrator/Inkscape)."""
        fig.savefig(FIG / png_name, dpi=160)
        fig.savefig(FIG / png_name.replace(".png", ".svg"))

    COL = ["#5f9ea0", "#8b0000", "#daa520", "#6a5acd", "#2e8b57"]  # the paper's palette for 1..5
    BG = "#e5e5e5"

    def style(ax, ylabel, xlabel):
        ax.set_facecolor(BG)
        ax.grid(axis="y", color="white", lw=1)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.tick_params(length=0)

    def bars(ax, groups, series, get, labels, hatches=None, width=0.15):
        """groups on the x axis, one coloured bar per series. get(g, s) -> {"mean": [10 values], "sd": [10 values]}.
        Bar = mean over the 10 datasets; error bar = std over individual reads (averaged over the datasets), like the paper."""
        for k, s in enumerate(series):
            xs = np.arange(len(groups)) + (k - (len(series) - 1) / 2) * width
            vals = [np.mean(get(g, s)["mean"]) for g in groups]
            errs = [np.mean(get(g, s)["sd"]) for g in groups]
            ax.bar(xs, vals, width, yerr=errs, color=COL[k % 5] if hatches is None else s[0], label=labels[k],
                   error_kw=dict(ecolor="black", lw=1, capsize=2), hatch=None if hatches is None else s[1],
                   edgecolor="white" if hatches is None else "#444", linewidth=0 if hatches is None else 0.6)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups)

    if (RES / "fig5b.json").exists():
        raw = json.loads((RES / "fig5b.json").read_text())
        d = {n: [{"mean": raw["mean"][n][i], "sd": raw["sd"][n][i]} for i in range(5)] for n in raw["mean"]}
        for name, groups, fn in [("fig5b_reproduction.png", ["MaxNW (paper metric)", "Random", "MinNW"], "(b) Insertion Technique"),
                                 ("fig5b_with_full_maxnw.png", ["MaxNW (paper metric)", "MaxNW (full payload)", "Random", "MinNW"], "(b) Insertion Technique")]:
            fig, ax = plt.subplots(figsize=(7.5 if len(groups) == 3 else 9.5, 3.6))
            bars(ax, groups, range(5), lambda g, s: d[g][s], [str(i) for i in range(1, 6)])
            style(ax, "NW Score Reduction (%)", fn)
            ax.set_xticklabels([g.replace("MaxNW (paper metric)", "MaxNW").replace(" (full payload)", "\n(full payload)") if len(groups) == 3
                                else g.replace(" (", "\n(") for g in groups])
            ax.legend(title="Fragment Size", ncol=2, frameon=False, loc="upper left", fontsize=8, title_fontsize=8)
            fig.tight_layout()
            save(fig, name)
            plt.close(fig)
        print("  drew fig5b_*.png")

    if (RES / "fig6.json").exists():
        d = json.loads((RES / "fig6.json").read_text())
        fig, axs = plt.subplots(2, 1, figsize=(7.5, 6.2))
        bars(axs[0], [str(s) for s in range(1, 6)], range(1, 6), lambda g, r: d["a"][str(r)][int(g) - 1],
             [str(r) for r in range(1, 6)])
        style(axs[0], "NW Score Reduction (%)", "(a) Fragment Size")
        axs[0].legend(title="Retention Position (key=50)", ncol=3, frameon=False, loc="upper left", fontsize=8, title_fontsize=8)
        axs[0].set_ylim(0, 4.5)
        keys = [0, 10, 20, 30, 40, 50]
        combos = [(COL[0], ".."), (COL[0], "//"), (COL[4], ".."), (COL[4], "//")]  # (size 1 | 5) x (retention 1 | 5)
        keyed = [(1, 1), (1, 5), (5, 1), (5, 5)]
        bars(axs[1], [("No Encryption" if k == 0 else f"key={k}") for k in keys], combos,
             lambda g, s: d["b"][f"{0 if g == 'No Encryption' else int(g[4:])}|{keyed[combos.index(s)][0]}|{keyed[combos.index(s)][1]}"],
             [f"size {a}, retention {b}" for a, b in keyed], hatches=True, width=0.18)
        style(axs[1], "NW Score Reduction (%)", "(b) Encryption Key")
        axs[1].legend(ncol=2, frameon=False, loc="upper left", fontsize=8)
        axs[1].set_ylim(0, 4.5)
        fig.tight_layout()
        save(fig, "fig6_reproduction.png")
        plt.close(fig)
        print("  drew fig6_reproduction.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "draw"
    if what == "5b":
        run_5b()
    elif what == "6":
        run_6()
    draw()
