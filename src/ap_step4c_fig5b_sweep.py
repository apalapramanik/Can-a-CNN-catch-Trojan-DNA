"""Step 4c: which setting reproduces paper Fig. 5b best? Sweep the scoring matrix and the payload format.

WHO DID WHAT
  Authors: Fig. 5b (the experiment) and the code settings (PAM250, payload 'First.Last:port'); the paper text says PAM10;
    their helper make_shorter_payloads (generate_dataset_new.py) turns 'domain.tld:port' into 'domain<port>tld'.
  Us: this sweep. The paper's numbers below were read off the figure by eye, so they are approximate (about +-0.5 for small
    values, +-1 for large ones); only large differences are meaningful.
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

CFG = json.loads((HERE / "ap_config.json").read_text())
PAPER = {  # Fig. 5b, fragment sizes 1..5, read off the figure
    "MaxNW": [1.2, 1.7, 2.1, 2.6, 3.1],
    "Random": [13.5, 8.9, 7.2, 7.0, 6.0],
    "MinNW": [19.5, 12.8, 10.0, 9.8, 8.3],
}


def load_reads():
    plasmids = [ln.split('",')[1].strip() for ln in (ROOT / CFG["paths"]["raw_plasmid_dna"]).read_text().splitlines() if '",' in ln]
    R = CFG["reads"]["read_size"]
    return [p[k * R:(k + 1) * R] for p in plasmids for k in range(len(p) // R) if set(p[k * R:(k + 1) * R]) <= set("ACGTN")]


def shorter(payload):
    """The authors' make_shorter_payloads: 'domain.tld:port' -> 'domain<port>tld'."""
    domain, rest = payload.split(".")
    tld, port = rest.split(":")
    return f"{domain}{port}{tld}"


def authors_payload(rng):
    """The authors' seed_data_generator.py payload, already rearranged by make_shorter_payloads:
    domain = 1-2 random lowercase letters, tld = 2 random letters, port = 1..99 -> 'domain<port>tld', e.g. 'xy42ab'."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    domain = "".join(rng.choice(list(letters), size=int(rng.integers(1, 3))))
    tld = "".join(rng.choice(list(letters), size=2))
    return f"{domain}{int(rng.integers(1, 100))}{tld}"


def run(matrix, reads, shorter_payload=False, key=60, retention=0, seed=1, payload_fn=None, tags=None):
    """Mean score reduction (%) per strategy and fragment size. Same payload for every strategy on a given read.
    payload_fn: function(rng) -> payload text (default: fake First.Last:port). tags: (start, end) tag letters (default ACTG/ACTG)."""
    rng = np.random.default_rng(seed)
    if tags:
        enc.START_TAG, enc.END_TAG = tags
    out = {"MaxNW": [], "Random": [], "MinNW": []}
    for size in range(1, 6):
        red = {k: [] for k in out}
        for host in reads:
            msg = payload_fn(rng) if payload_fn else gr.fake_payload(rng)
            if shorter_payload:
                msg = shorter(msg)
            frags = enc.make_fragments(msg, size, key, retention)
            base = sc.ungapped_score(host, host, matrix)
            for name, seq in (("MaxNW", gr.place_greedy(host, frags, matrix, True)[0]),
                              ("Random", gr.place_random(host, frags, rng)[0]),
                              ("MinNW", gr.place_greedy(host, frags, matrix, False)[0])):
                red[name].append(100 * (base - sc.ungapped_score(host, seq, matrix)) / base)
        for k in out:
            out[k].append(float(np.mean(red[k])))
    enc.START_TAG = enc.END_TAG = "ACTG"  # restore the default
    return out


def show(title, res):
    err = np.mean([abs(a - b) for k in PAPER for a, b in zip(res[k], PAPER[k])])
    print(f"\n{title}   (mean |ours - paper| = {err:.2f} points)")
    print("   size:     " + "".join(f"{s:>8d}" for s in range(1, 6)))
    for k in ("MaxNW", "Random", "MinNW"):
        print(f"   {k:7s}  ours" + "".join(f"{v:8.2f}" for v in res[k]))
        print(f"   {'':7s} paper" + "".join(f"{v:8.1f}" for v in PAPER[k]))
    return err


if __name__ == "__main__":
    N = 200
    reads = load_reads()[:N]
    mats = {name: sc.load_matrix(path) for name, path in
            [("PAM250 (authors' code)", ROOT / "authors_repo/trojan_attack/PAM250"),
             ("PAM30", ROOT / "matrices/PAM30"),
             ("PAM10 (paper text)", ROOT / "matrices/PAM10")]}
    for name, M in mats.items():
        show(f"{name}, payload as-is", run(M, reads))
    show("PAM10, shorter payload (domain+port+tld)", run(mats["PAM10 (paper text)"], reads, shorter_payload=True))
    print("\n==== now with the authors' REAL experiment settings: tags TCATG/GTTGG, payloads like 'xy42ab' ====")
    AUTH = dict(payload_fn=authors_payload, tags=("TCATG", "GTTGG"))
    for name, M in mats.items():
        show(f"{name}, authors' tags + payloads", run(M, reads, **AUTH))
