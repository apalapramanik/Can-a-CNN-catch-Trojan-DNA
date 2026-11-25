"""Step 14: ap_step13_real_decoder_check.py scaled to the FULL dataset (all 1000 host/payload pairs in
fragment_len_5/retention_pos_0/encryption_key_0/dataset_0, instead of 20), plus box plots showing how much
the outcome varies READ TO READ (not just the mean).

WHO DID WHAT
  User: asked to scale ap_step13 to the full dataset and add box plots for the spread/std, after
    confirming scope (all 1000 hosts, one dataset), box content (per-read success rate, one box per N),
    and the repeats/runtime trade-off (keep all 7 machines, fewer repeats per trial).
  Us: this script. REPEATS reduced from 40 to 10 (fewer noisy trials per read) to keep runtime reasonable
    at 1000 reads; Ns trimmed to the range where real variation happens (1,3,5,7,9 -- by N=9 every real
    machine was already saturated at ~100%% in ap_step13, so larger N adds no new information).

Per-read metric (what each box in the box plot is one point of):
  - Malware: for one host/payload pair, the FRACTION of the REPEATS noisy trials where the real decoder
    (read_payload_robust) returned the exact true payload. A read whose payload sits near a fragile tag
    position will have a lower rate than one that doesn't -- this is what "read-to-read variability" means.
  - Legitimate: for one host, the MEAN letter accuracy (vs. the true host) over the REPEATS trials.

Usage: python3 ap_step14_full_dataset_boxplot.py   (several minutes)
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step3_encode as enc
import ap_step10_consensus_tradeoff as c

enc.START_TAG, enc.END_TAG = "TCATG", "GTTGG"

MACHINES = {
    "Nanopore (Old)": 0.0788,
    "Nanopore (R10.4.1)": 0.0316,
    "MiniSeq": 0.00613,
    "MiSeq": 0.00473,
    "NextSeq 500": 0.00429,
    "HiSeq 2500": 0.00112,
    "NovaSeq 6000": 0.00109,
}

rng = np.random.default_rng(0)
N_READS = 1000   # ALL hosts in the dataset, not just 20
REPEATS = 10      # reduced from 40, to keep the full-dataset run fast
Ns = [1, 3, 5, 7, 9]

t0 = time.time()
hosts, injected = c.load_pairs(size=5, ret=0, key=0, ds=0, scenario="best", n=N_READS)
base = ROOT / "data/datasets/fragment_len_5/retention_pos_0/encryption_key_0/dataset_0"
payloads = base.joinpath("payloads.txt").read_text().split("\n")[:-1][:N_READS]
print(f"loaded {len(hosts)} host/payload pairs in {time.time() - t0:.1f}s")

host_diffs = []  # pre-compute the payload footprint for each read once
for h, j in zip(hosts, injected):
    host_diffs.append(c.diff_positions(h, j))


def per_read_malware_rate(N, p_err):
    """Returns an array of length N_READS: for each read, the fraction of REPEATS trials that decoded
    to the exact true payload."""
    rates = np.zeros(len(hosts))
    for i, (j, payload) in enumerate(zip(injected, payloads)):
        jc = np.array(list(j))
        hits = 0
        for _ in range(REPEATS):
            copies = np.stack([c.noisy(jc, p_err, rng) for _ in range(N)])
            cons = "".join(c.majority_consensus(copies))
            hits += (enc.read_payload_robust(cons, key=0, retention=0) == payload)
        rates[i] = hits / REPEATS
    return rates


def per_read_legit_rate(N, p_err):
    """Returns an array of length N_READS: for each read, mean letter accuracy (vs. true host) at the
    payload-footprint positions, over REPEATS trials."""
    rates = np.zeros(len(hosts))
    for i, (h, reg) in enumerate(zip(hosts, host_diffs)):
        hr = np.array(list(h))[reg]
        vals = []
        for _ in range(REPEATS):
            copies = np.stack([c.noisy(hr, p_err, rng) for _ in range(N)])
            cons = c.majority_consensus(copies)
            vals.append((cons == hr).mean())
        rates[i] = np.mean(vals)
    return rates


malware_per_read = {}   # (machine, N) -> array of N_READS per-read rates
legit_per_read = {}
for name, p_err in MACHINES.items():
    t1 = time.time()
    for N in Ns:
        malware_per_read[(name, N)] = per_read_malware_rate(N, p_err)
        legit_per_read[(name, N)] = per_read_legit_rate(N, p_err)
    print(f"{name:20s} done in {time.time() - t1:.0f}s "
          f"(mean malware-blocked%% at N=1: {100 * (1 - malware_per_read[(name, Ns[0])].mean()):.1f}, "
          f"std across reads: {100 * malware_per_read[(name, Ns[0])].std():.1f})")

# ---- save summary (mean + std per machine/N, for both metrics)
summary = {"Ns": Ns, "machines": MACHINES, "n_reads": N_READS, "repeats": REPEATS,
           "malware_blocked_mean": {}, "malware_blocked_std": {},
           "legit_quality_mean": {}, "legit_quality_std": {}}
for name in MACHINES:
    summary["malware_blocked_mean"][name] = {str(N): float(100 * (1 - malware_per_read[(name, N)].mean())) for N in Ns}
    summary["malware_blocked_std"][name] = {str(N): float(100 * malware_per_read[(name, N)].std()) for N in Ns}
    summary["legit_quality_mean"][name] = {str(N): float(100 * legit_per_read[(name, N)].mean()) for N in Ns}
    summary["legit_quality_std"][name] = {str(N): float(100 * legit_per_read[(name, N)].std()) for N in Ns}

out = ROOT / "results/step14_full_dataset_boxplot.json"
out.write_text(json.dumps(summary, indent=2))
print(f"saved {out}  (total time {time.time() - t0:.0f}s)")

# ---- plot 1: the mean lines (like step13) but now with a shaded +-1 std band, full 1000-read dataset
BG = "#EAEAF2"
COL = ["#C44E52", "#DD8452", "#E8A33D", "#55A868", "#4C72B0", "#8172B2", "#937860"]


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(color="white", lw=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(11, 7.5))
for col, (name, p_err) in zip(COL, MACHINES.items()):
    bm = np.array([summary["malware_blocked_mean"][name][str(N)] for N in Ns])
    bs = np.array([summary["malware_blocked_std"][name][str(N)] for N in Ns])
    lm = np.array([summary["legit_quality_mean"][name][str(N)] for N in Ns])
    ls = np.array([summary["legit_quality_std"][name][str(N)] for N in Ns])
    ax.plot(Ns, bm, "o-", color=col, lw=2.5, ms=7, label=f"{name} ({100 * p_err:.2f}% Error) -- Malware Blocked")
    ax.fill_between(Ns, bm - bs, bm + bs, color=col, alpha=0.12)
    ax.plot(Ns, lm, "o:", color=col, lw=2.5, ms=7, alpha=0.6, label=f"{name} -- Legitimate Decode Quality")
ax.set_xlabel("No Of Copies", fontsize=16)
ax.set_ylabel("Percentage (%)", fontsize=16)
ax.legend(fontsize=8, ncol=2, loc="center right")
style(ax)
fig.tight_layout()
for e in ("png", "svg"):
    fig.savefig(ROOT / f"figures/step14_full_dataset_lines.{e}", dpi=150, bbox_inches="tight")
print("saved figures/step14_full_dataset_lines.png/.svg")

# ---- plot 2: box plots, read-to-read variability, one figure per metric, one subplot per machine
def box_figure(per_read_dict, title_suffix, fname):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    axes = axes.flatten()
    for ax_i, (name, p_err) in zip(axes, MACHINES.items()):
        data = [100 * per_read_dict[(name, N)] for N in Ns]
        bp = ax_i.boxplot(data, positions=Ns, widths=1.2, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.6)
        ax_i.set_title(f"{name} ({100 * p_err:.2f}% Error)", fontsize=12)
        ax_i.set_xlabel("No Of Copies", fontsize=11)
        style(ax_i)
    axes[0].set_ylabel("Percentage (%)", fontsize=13)
    axes[4].set_ylabel("Percentage (%)", fontsize=13)
    for ax_i in axes[len(MACHINES):]:
        ax_i.axis("off")
    fig.suptitle(f"Read-To-Read Variability: {title_suffix}", fontsize=15)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(ROOT / f"figures/{fname}.{e}", dpi=150, bbox_inches="tight")
    print(f"saved figures/{fname}.png/.svg")


box_figure(malware_per_read, "Malware Decode Success Per Read (% Of Repeats)", "step14_box_malware")
box_figure(legit_per_read, "Legitimate Decode Quality Per Read (Letter Accuracy %)", "step14_box_legit")
