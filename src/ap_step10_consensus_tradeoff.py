"""Step 10: small, separate simulation of coverage-depth majority-vote consensus decoding, and the
N (copies) / f (fraction malicious) trade-off.

WHO DID WHAT
  Prof.: the idea that real sequencing reconstructs a region from many independent reads/copies by a
    per-base majority vote (how real coverage-depth consensus calling works), and the question of whether
    a higher number of copies N makes it MORE likely that a malware edit survives into the final consensus,
    if the attacker can get the edit into a majority of the copies -- versus fewer copies being harder for
    the attacker to win, but also worse at correcting ordinary sequencing noise.
  Us: this toy simulation, built on 20 already-generated host/injected read pairs (fragment size 5,
    retention 0, key 0, dataset 0, MaxNW) -- NOT the full 250-dataset pipeline, just enough pairs to see
    the effect. New here: noisy_copy (simulates per-base sequencing error), majority_consensus (vectorised
    per-position vote), run_trial (mixes host and injected copies at a chosen fraction f).

Mechanism simulated
  For one host read and its already-injected twin: find the positions where they actually differ (the real
  payload footprint, whatever the fragment/retention settings did). Make N "sequenced copies": a fraction f
  of them are the injected version, the rest are the clean host; every copy also gets independent random
  per-base noise p_err (ordinary sequencing error). Take the per-position majority vote across the N noisy
  copies -> the consensus. Check whether the consensus, at the payload's own positions, matches the
  malicious payload or the clean host.

Usage: python3 ap_step10_consensus_tradeoff.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASES = np.array(list("ACGT"))


def load_pairs(size=5, ret=0, key=0, ds=0, scenario="best", n=20):
    base = ROOT / f"data/datasets/fragment_len_{size}/retention_pos_{ret}/encryption_key_{key}/dataset_{ds}"
    names = {"best": "nw_best_greedy_trojan_insertion_dataset.txt",
             "worst": "nw_worst_greedy_trojan_insertion_dataset.txt",
             "random": "random_trojan_insertion_dataset.txt"}
    hosts = base.joinpath("non_trojan_dataset.txt").read_text().split("\n")[:-1][:n]
    injected = base.joinpath(names[scenario]).read_text().split("\n")[:-1][:n]
    return hosts, injected


def diff_positions(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def noisy(chars, p_err, rng):
    """chars: 1-D array of single-character bases. Returns a noisy copy (independent per-base error)."""
    out = chars.copy()
    flip = rng.random(len(chars)) < p_err
    if flip.any():
        out[flip] = rng.choice(BASES, flip.sum())
    return out


def majority_consensus(copies):
    """copies: (N, L) char array. Vectorised per-position majority vote over the 4 bases."""
    counts = np.stack([(copies == b).sum(axis=0) for b in BASES])  # (4, L)
    return BASES[np.argmax(counts, axis=0)]


def run_trial_count(host_region, inj_region, N, n_mal, p_err, rng):
    """n_mal: the EXACT number of malicious copies (not a fraction) -- an attacker who can only produce
    a fixed number of tainted copies, regardless of how many total copies (N) end up sequenced."""
    n_mal = min(n_mal, N)
    sources = [inj_region] * n_mal + [host_region] * (N - n_mal)
    copies = np.stack([noisy(s, p_err, rng) for s in sources])
    return majority_consensus(copies)


def run_trial(host_region, inj_region, N, f, p_err, rng):
    """f: the FRACTION of copies that are malicious (scales with N). See run_trial_count for a fixed count."""
    return run_trial_count(host_region, inj_region, N, int(round(f * N)), p_err, rng)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    hosts, injected = load_pairs()

    print("=" * 72)
    print("PART 1: one worked example -- single position, N=5 copies, NO sequencing noise")
    print("=" * 72)
    host, inj = hosts[0], injected[0]
    region = diff_positions(host, inj)
    pos = region[0]
    print(f"payload changes {len(region)} of 1000 positions; using position {pos}: "
          f"host={host[pos]!r}  payload={inj[pos]!r}")
    for n_mal in range(6):
        votes = [inj[pos]] * n_mal + [host[pos]] * (5 - n_mal)
        vals, counts = np.unique(votes, return_counts=True)
        winner = vals[np.argmax(counts)]
        tag = "ATTACK WINS" if winner == inj[pos] else "host recovered"
        print(f"  {n_mal} of 5 copies malicious -> votes {votes} -> consensus {winner!r}  ({tag})")

    print()
    print("=" * 72)
    print("PART 2: sweep N and f, p_err=1%% noise per copy, 20 read pairs x 50 repeats each")
    print("        value = %% of the PAYLOAD'S OWN positions that survive into the consensus")
    print("=" * 72)
    Ns = [5, 11, 21, 51, 101]
    fs = [0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0]
    P_ERR, REPEATS = 0.01, 50
    results = {}
    print("N\\f   " + "  ".join(f"{f:.2f}" for f in fs))
    for N in Ns:
        row = []
        for f in fs:
            survive = []
            for h, j in zip(hosts, injected):
                reg = diff_positions(h, j)
                hr, jr = np.array(list(h))[reg], np.array(list(j))[reg]
                for _ in range(REPEATS):
                    cons = run_trial(hr, jr, N, f, P_ERR, rng)
                    survive.append((cons == jr).mean())
            row.append(float(np.mean(survive)))
        results[N] = dict(zip(fs, row))
        print(f"{N:4d}  " + "  ".join(f"{v*100:5.1f}%" for v in row))

    print()
    print("=" * 72)
    print("PART 3: does LOWER N correct ORDINARY sequencing noise WORSE? (f=0, no attacker)")
    print("        consensus error rate vs. the TRUE host, over the full 1000-base read")
    print("        (N=1 means no voting at all: the raw noisy read is the 'consensus')")
    print("=" * 72)
    Ns_noise = [1, 2, 3, 5, 7, 9, 11, 15, 21, 51, 101]
    p_errs = [0.01, 0.05, 0.15]
    REPEATS2 = 30
    noise_results = {}
    print("N\\p_err  " + "  ".join(f"{p * 100:4.0f}%" for p in p_errs))
    for N in Ns_noise:
        row = []
        for p_err in p_errs:
            errs = []
            for h in hosts:
                hc = np.array(list(h))
                for _ in range(REPEATS2):
                    cons = run_trial(hc, hc, N, 0.0, p_err, rng)  # f=0: every copy is the clean host
                    errs.append((cons != hc).mean())
            row.append(float(np.mean(errs)))
        noise_results[N] = dict(zip(p_errs, row))
        print(f"{N:5d}    " + "  ".join(f"{v * 100:5.3f}%" for v in row))

    print()
    print("=" * 72)
    print("PART 4: FIXED NUMBER of malicious copies (not a fraction) -- an attacker who can only")
    print("        produce a set count of tainted copies, however many total copies (N) get sequenced.")
    print("        Low N -> that fixed count is a BIGGER share of the vote -> easier to win.")
    print("=" * 72)
    counts = [1, 2, 3, 5, 10]
    Ns_count = [1, 2, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101, 151, 201]
    count_results = {}
    print("count\\N  " + "  ".join(f"{N:4d}" for N in Ns_count))
    for m in counts:
        row = []
        for N in Ns_count:
            if m > N:
                row.append(None)
                continue
            survive = []
            for h, j in zip(hosts, injected):
                reg = diff_positions(h, j)
                hr, jr = np.array(list(h))[reg], np.array(list(j))[reg]
                for _ in range(REPEATS):
                    cons = run_trial_count(hr, jr, N, m, P_ERR, rng)
                    survive.append((cons == jr).mean())
            row.append(float(np.mean(survive)))
        count_results[m] = dict(zip(Ns_count, row))
        print(f"{m:5d}    " + "  ".join("  -- " if v is None else f"{v * 100:4.0f}%" for v in row))

    out = ROOT / "results/step10_consensus_tradeoff.json"
    out.write_text(json.dumps({"sweep_payload_survival": {str(n): r for n, r in results.items()},
                                "noise_correction_vs_N": {str(n): r for n, r in noise_results.items()},
                                "fixed_count_vs_N": {str(m): {str(n): v for n, v in r.items() if v is not None}
                                                     for m, r in count_results.items()},
                                "p_err_vote": P_ERR}, indent=2))
    print(f"\nsaved sweep to {out}")
