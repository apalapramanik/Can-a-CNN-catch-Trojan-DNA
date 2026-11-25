"""Tests and the Fig. 5b comparison for ap_step4b_greedy.py."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step3_encode as enc
import ap_step4_score as sc
import ap_step4b_greedy as gr

CFG = json.loads((HERE / "ap_config.json").read_text())
M = sc.load_matrix(ROOT / "authors_repo/trojan_attack/PAM250")

# real reads (same cutting rule as step 2)
plasmids = [ln.split('",')[1].strip() for ln in (ROOT / CFG["paths"]["raw_plasmid_dna"]).read_text().splitlines() if '",' in ln]
R = CFG["reads"]["read_size"]
reads = [p[k * R:(k + 1) * R] for p in plasmids for k in range(len(p) // R) if set(p[k * R:(k + 1) * R]) <= set("ACGTN")]

# ---- load the authors' greedy placer. Their file needs the library nwalign3, which we do not install, so we give it a
# stand-in whose score_alignment is OUR tested scorer. This tests their PLACEMENT logic, not their scoring.
stub = types.ModuleType("nwalign3")
stub.score_alignment = lambda a, b, gap_open, gap_extend, matrix: sc.ungapped_score(a, b, M)
sys.modules["nwalign3"] = stub
spec = importlib.util.spec_from_file_location("their_nw", ROOT / "authors_repo/trojan_attack/nw_insertion_new.py")
their = importlib.util.module_from_spec(spec)
spec.loader.exec_module(their)
nwi = their.NW_INSERTION()

rng = np.random.default_rng(1)

print("1) Our greedy placement == the authors' greedy placement (same injected read, best and worst)")
checked = 0
for size, ret in [(5, 0), (2, 1), (1, 0)]:
    for k in range(2):
        host = reads[k * 37]
        msg = gr.fake_payload(rng)
        frags = enc.make_fragments(msg, size, 60, ret)
        their_frags = [f.replace("*", "X") for f in frags]  # authors mark retention slots with 'X'
        for best in (True, False):
            mine, _ = gr.place_greedy(host, frags, M, best=best)
            theirs, _, _ = nwi.get_nw_based_insertions_greedy(host, their_frags, return_best=best)
            assert mine == theirs, (size, ret, best)
            checked += 1
print(f"   identical on {checked} cases (fragment size 1/2/5, retention 0/1, best and worst)")

print("2) Greedy output really contains the message (decode each fragment at the position where it was placed)")
host, msg = reads[5], "Ann.Cole:1234"
for size in (1, 2, 5):
    frags = enc.make_fragments(msg, size, 60, 0)
    inj, starts = gr.place_greedy(host, frags, M, best=True)
    parts = dict(enc.decode_fragment(inj[i + 4:i + len(f) - 4], 60) for f, i in zip(frags, starts))  # strip the 4-letter tags
    got = "".join(parts[k] for k in sorted(parts))
    assert got == msg
    print(f"   fragment size {size}: decoded {got!r}  OK")
nat = [r.count(enc.START_TAG) for r in reads]
print(f"   NOTE: the tag {enc.START_TAG} already occurs naturally {np.mean(nat):.1f} times per clean read ({100 * np.mean([c > 0 for c in nat]):.0f}% of reads),")
print("   so a decoder that just scans for the tag gets garbage on clean reads too (the authors' decoder does this as well).")

print("3) Paper Fig. 5b: NW score reduction (%) by fragment size and insertion technique (retention 0, key 60, 300 reads)")
paper = {"MaxNW": [1.2, 1.7, 2.1, 2.6, 3.1], "Random": [13.5, 8.9, 7.2, 7.0, 6.0], "MinNW": [19.5, 12.8, 10.0, 9.8, 8.3]}  # read off the figure, approx.
N = 300
print("   size |        MaxNW        |       Random        |        MinNW")
print("        |  ours  (paper~)     |  ours  (paper~)     |  ours  (paper~)")
for size in range(1, 6):
    red = {"MaxNW": [], "Random": [], "MinNW": []}
    for host in reads[:N]:
        frags = enc.make_fragments(gr.fake_payload(rng), size, 60, 0)
        base = sc.ungapped_score(host, host, M)
        for name, seq in (("MaxNW", gr.place_greedy(host, frags, M, True)[0]),
                          ("Random", gr.place_random(host, frags, rng)[0]),
                          ("MinNW", gr.place_greedy(host, frags, M, False)[0])):
            red[name].append(100 * (base - sc.ungapped_score(host, seq, M)) / base)
    row = "   " + f"{size:4d} |"
    for name in ("MaxNW", "Random", "MinNW"):
        row += f" {np.mean(red[name]):5.2f}  ({paper[name][size - 1]:5.1f})    |"
    print(row)
