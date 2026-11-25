"""Tests and a demo for ap_step4_score.py."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step3_encode as enc
import ap_step4_score as sc

CFG = json.loads((HERE / "ap_config.json").read_text())
M = sc.load_matrix(ROOT / "authors_repo/trojan_attack/PAM250")

print("1) Does our scorer reproduce the examples in the nwalign3 documentation? (gap_open=-5, gap_extend=-2)")
for a, b, expected in [("CEELECANTH", "-PELICAN--", 11), ("CEELE-CANTH", "-PEL-ICAN--", 6)]:
    got = sc.score_alignment(a, b, M)
    print(f"   {a} / {b}: docs say {expected}, ours {got}")
    assert got == expected

# real reads (same cutting rule as step 2)
plasmids = [ln.split('",')[1].strip() for ln in (ROOT / CFG["paths"]["raw_plasmid_dna"]).read_text().splitlines() if '",' in ln]
R = CFG["reads"]["read_size"]
reads = [p[k * R:(k + 1) * R] for p in plasmids for k in range(len(p) // R) if set(p[k * R:(k + 1) * R]) <= set("ACGTN")]

print("2) Fast scorer equals brute force (inject for real, then score the whole read)")
rng = np.random.default_rng(0)
payload = enc.make_payload("john.smith:4821", 5, 60, 1)
host = reads[0]
fast = sc.scores_for_all_positions(host, payload, M)
for i in rng.integers(len(fast), size=20):
    assert fast[i] == sc.ungapped_score(host, enc.inject(host, payload, int(i)), M)
print(f"   OK on 20 random positions ({len(fast)} possible positions, payload length {len(payload)})")

print("3) Best / worst / random injection spot on 200 reads (one block, fragment size 5, key 60, retention 1)")
self_score = sc.ungapped_score(host, host, M)
red = {"MaxNW": [], "Random": [], "MinNW": []}
for host in reads[:200]:
    s = sc.scores_for_all_positions(host, payload, M)
    base = sc.ungapped_score(host, host, M)
    for name, i in sc.pick_positions(s, rng).items():
        red[name].append(100 * (base - s[i]) / base)
for name in ("MaxNW", "Random", "MinNW"):
    print(f"   {name:7s} score reduction: mean {np.mean(red[name]):5.2f}%   (std {np.std(red[name]):.2f})")
assert np.mean(red["MaxNW"]) < np.mean(red["Random"]) < np.mean(red["MinNW"])
