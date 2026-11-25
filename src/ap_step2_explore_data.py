"""Step 2: look at the raw plasmid data before building anything on it."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent  # trojan_reimplementation/
CFG = json.loads((HERE / "ap_config.json").read_text())
DATA = ROOT / CFG["paths"]["raw_plasmid_dna"]

# Each line looks like:  1, 107251,"> Addgene NGS Result",CCCC...
# The DNA is everything after the '",' -- same trick the authors use.
plasmids = []
for line in DATA.read_text().splitlines():
    if '",' not in line:
        continue
    plasmids.append(line.split('",')[1].strip())

print("plasmids:", len(plasmids))

lengths = [len(p) for p in plasmids]
print("length min / mean / max:", min(lengths), round(sum(lengths) / len(lengths)), max(lengths))

letters = Counter("".join(plasmids))
total = sum(letters.values())
print("total letters:", total)
for ch, n in letters.most_common():
    print(f"  {ch}: {n:>9}  {100 * n / total:5.2f}%")

# Cut non-overlapping 1000-letter reads, skipping any with letters outside ACGTN
# (this mirrors the authors' make_reads).
READ = CFG["reads"]["read_size"]
reads = []
for p in plasmids:
    for k in range(len(p) // READ):
        r = p[k * READ:(k + 1) * READ]
        if set(r) <= set("ACGTN"):
            reads.append(r)
print("usable 1000-letter reads:", len(reads))
print("first read, first 80 letters:", reads[0][:80])

# Sanity check: measured numbers must agree with ap_config.json
assert len(plasmids) == CFG["dataset_stats"]["num_sequences"]
assert total == CFG["dataset_stats"]["total_letters"]
assert len(reads) == CFG["reads"]["num_usable_reads"]
print("config matches data: OK")
