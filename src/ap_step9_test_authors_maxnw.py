"""Step 9a: run the AUTHORS' MaxNW/MinNW code (nw_insertion_new.py + encode_decode.py) with the REAL nwalign3 library.

WHO DID WHAT
  Authors: nw_insertion_new.py (greedy placement, scoring via nwalign3.score_alignment), encode_decode.py (fragments).
  Us: this test. Earlier (Step 4) we only ran their placement with a stand-in scorer that was our own code. Here nothing is replaced.
  The authors' files are imported unchanged (authors_repo is never edited).

Checks
  1) real nwalign3.score_alignment == our ungapped scorer (PAM250 and PAM10) on random read pairs.
  2) their greedy MaxNW / MinNW (matrix = PAM10 file) == the injected reads we generated in data/datasets (same hosts, same payloads).
  3) what their DEFAULT setting (matrix 'PAM250') gives instead.
  4) the Fig. 5b number as their code computes it (np.max of running scores = score after the FIRST fragment) vs the true final score.

Run with the Python 3.8 env that has nwalign3 built:   ~/venvs/ap_nw8/bin/python -I ap_step9_test_authors_maxnw.py [n_lines]
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "authors_repo/trojan_attack"))
import nwalign3 as nw
import ap_step4_score as sc
from encode_decode import EncodeDecode            # authors' file, unchanged
from nw_insertion_new import NW_INSERTION          # authors' file, unchanged

# the library opens the matrix by file name relative to the working folder; the authors run from trojan_attack/ where PAM250 sits
os.chdir(ROOT / "authors_repo/trojan_attack")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PAM10 = str(ROOT / "matrices/PAM10")
M10 = sc.load_matrix(ROOT / "matrices/PAM10")
M250 = sc.load_matrix(ROOT / "authors_repo/trojan_attack/PAM250")
print("nwalign3 from:", nw.__file__)

# ---- 1) scorer
rng = np.random.default_rng(0)
bad = 0
for _ in range(300):
    a = "".join(rng.choice(list("ACGTN"), 1000))
    b = "".join(rng.choice(list("ACGTN"), 1000))
    for name, path, M in (("PAM250", "PAM250", M250), ("PAM10", PAM10, M10)):
        lib = nw.score_alignment(a, b, gap_open=-2, gap_extend=-2, matrix=path)
        mine = sc.ungapped_score(a, b, M)
        if abs(lib - mine) > 1e-6:
            bad += 1
print("1) library score == our ungapped score on 300 random pairs x 2 matrices:", "ALL EQUAL" if bad == 0 else f"{bad} MISMATCHES")

# ---- 2) and 3) placement
def load_lines(p):
    return (p).read_text().split("\n")[:-1]


def run_case(size, ret, key, ds, matrix):
    base = ROOT / f"data/datasets/fragment_len_{size}/retention_pos_{ret}/encryption_key_{key}/dataset_{ds}"
    hosts = load_lines(base / "non_trojan_dataset.txt")[:N]
    pays = load_lines(base / "payloads.txt")[:N]
    ours_best = load_lines(base / "nw_best_greedy_trojan_insertion_dataset.txt")[:N]
    ours_worst = load_lines(base / "nw_worst_greedy_trojan_insertion_dataset.txt")[:N]
    ed = EncodeDecode("TCATG", "GTTGG", ret)
    nwi = NW_INSERTION(matrix=matrix)
    same_b = same_w = 0
    first, final = [], []
    for h, p, ob, ow in zip(hosts, pays, ours_best, ours_worst):
        frags = ed.make_fragments(p, size, key > 0, key)
        sb, score_b, _ = nwi.get_nw_based_insertions_greedy(h, frags, return_best=True)
        sw, _, _ = nwi.get_nw_based_insertions_greedy(h, frags, return_best=False)
        same_b += sb == ob
        same_w += sw == ow
        first.append(score_b)
        final.append(nwi.get_nw_score(h, sb))
    return same_b, same_w, float(np.mean(first)), float(np.mean(final)), float(np.mean([nwi.get_nw_score(h, s) for h, s in zip(hosts, ours_best)])), hosts


print(f"\n2) their greedy code + real nwalign3 + PAM10 file vs our generated reads ({N} lines per case)")
print("   case                      MaxNW identical   MinNW identical")
for size, ret, key in [(1, 0, 0), (5, 0, 0), (1, 3, 0), (5, 5, 0), (1, 0, 20)]:
    sb, sw, *_ = run_case(size, ret, key, 0, PAM10)
    print(f"   size {size} ret {ret} key {key:2d}        {sb:3d}/{N}            {sw:3d}/{N}")

print(f"\n3) their DEFAULT matrix (PAM250) vs our PAM10 reads: how often do they place the payload identically? ({N} lines)")
for size, ret, key in [(1, 0, 0), (5, 0, 0)]:
    sb, sw, *_ = run_case(size, ret, key, 0, "PAM250")
    print(f"   size {size} ret {ret} key {key}: MaxNW identical {sb}/{N}, MinNW identical {sw}/{N}")

print(f"\n4) Fig. 5b metric by their code (PAM10): mean score AFTER FIRST fragment (np.max) vs mean of the FINISHED read's score, as % of the host's self-score")
for size in (1, 2, 3, 4, 5):
    _, _, first, final, _, hosts = run_case(size, 0, 0, 0, PAM10)
    selfs = np.mean([sc.ungapped_score(h, h, M10) for h in hosts])
    print(f"   size {size}: reported (np.max) {100 * (selfs - first) / selfs:5.2f}%   true final {100 * (selfs - final) / selfs:5.2f}%")
