"""Checks on the files written by ap_step4d_generate_datasets.py (run it after the generator finishes)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step3_encode as enc
import ap_step4_score as sc
import ap_step4d_generate_datasets as gen

M = gen.M
hosts, payloads = gen.seed_data()
FILES = gen.NAMES
folders = sorted((ROOT / "data/datasets").glob("fragment_len_*/retention_pos_*/encryption_key_*"))
print(f"{len(folders)} settings found, {sum(len(list(f.glob('dataset_*'))) for f in folders)} datasets in total")

# 1) every file: 1000 lines of exactly 1000 letters from ACGTN
bad = 0
for f in folders:
    for ds in f.glob("dataset_*"):
        for fn in FILES.values():
            lines = (ds / fn).read_text().split("\n")[:-1]
            bad += len(lines) != 1000 or any(len(l) != 1000 or set(l) - set("ACGTN") for l in lines)
print("1) all files have 1000 lines x 1000 letters, alphabet ACGTN:", "OK" if bad == 0 else f"{bad} PROBLEMS")
assert bad == 0

# 2) clean file == the original hosts in the recorded order; injected reads differ from their host in at most (non-'*' payload letters)
checked = 0
for f in folders:
    size = int(f.parent.parent.name.split("_")[-1]); ret = int(f.parent.name.split("_")[-1]); key = int(f.name.split("_")[-1])
    ds = f / "dataset_0"
    meta = [l.split(",") for l in (ds / "decodes.csv").read_text().split("\n")[1:-1]]
    clean = (ds / FILES["clean"]).read_text().split("\n")[:-1]
    inj = {n: (ds / FILES[n]).read_text().split("\n")[:-1] for n in ("random", "best", "worst")}
    for line in range(0, 1000, 97):
        _, h, payload = int(meta[line][0]), int(meta[line][1]), meta[line][2]
        assert clean[line] == hosts[h], (f, line)
        letters = sum(c != "*" for fr in enc.make_fragments(payload, size, key, ret) for c in fr)
        for n in inj:
            changed = sum(a != b for a, b in zip(clean[line], inj[n][line]))
            assert changed <= letters, (f, n, line, changed, letters)
            checked += 1
print(f"2) clean file matches the hosts, and no injected read changes more letters than its payload has: OK ({checked} reads checked)")

# 3) similarity ordering MaxNW < Random < MinNW (score reduction of the whole read), averaged over 100 reads of dataset_0
print("3) mean score reduction (%) of the finished read, 100 reads per setting (dataset_0):")
print("   setting (size/ret/key)   best(MaxNW)  random   worst(MinNW)")
for f in folders:
    ds = f / "dataset_0"
    clean = (ds / FILES["clean"]).read_text().split("\n")[:-1][:100]
    red = {}
    for n in ("best", "random", "worst"):
        inj = (ds / FILES[n]).read_text().split("\n")[:-1][:100]
        red[n] = np.mean([100 * (sc.ungapped_score(c, c, M) - sc.ungapped_score(c, i, M)) / sc.ungapped_score(c, c, M) for c, i in zip(clean, inj)])
    assert red["best"] < red["random"] < red["worst"], (f, red)
    tag = "/".join(p.split("_")[-1] for p in (f.parent.parent.name, f.parent.name, f.name))
    print(f"   {tag:22s}   {red['best']:6.2f}      {red['random']:6.2f}    {red['worst']:6.2f}")
print("   ordering best < random < worst holds in every setting: OK")

# 4) how many injected reads carry a recoverable payload (smart decoder), over all datasets
tot = ok = 0
for f in folders:
    for ds in f.glob("dataset_*"):
        for row in (ds / "decodes.csv").read_text().split("\n")[1:-1]:
            v = row.split(",")
            ok += sum(int(v[i]) for i in (4, 6, 8)); tot += 3
print(f"4) smart Trojan recovers the payload from {100 * ok / tot:.1f}% of all {tot} injected reads")
