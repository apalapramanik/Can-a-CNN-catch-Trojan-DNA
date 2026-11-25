"""Step 4d: write the clean + injected datasets to files (what the CNN will train on).

WHO DID WHAT
  Authors: the dataset design and folder layout (generate_dataset_new.py): per (fragment size, retention, key) folder, 10 datasets,
    1000 hosts each, files for clean / random / greedy-best / greedy-worst. Their settings: tags TCATG/GTTGG, payloads like 'xy42ab'.
  Us: our own generator on top of the Step 3-4 code (no nwalign3 / names / Docker needed), the same file names, plus two extras:
    payloads.txt (which payload went into which line) and decodes.csv (does a tag-scanning Trojan recover the payload from each injected
    read? Tag collisions can break it, see Step 3/4b). We skip the authors' slow 'straight forward' permutation strategy (bs/ws).

Output:  data/datasets/fragment_len_F/retention_pos_R/encryption_key_K/dataset_D/
    non_trojan_dataset.txt                       1000 clean reads, one per line
    random_trojan_insertion_dataset.txt          same hosts, payload inserted at random positions (fragments in order)
    nw_best_greedy_trojan_insertion_dataset.txt  same hosts, MaxNW greedy placement
    nw_worst_greedy_trojan_insertion_dataset.txt same hosts, MinNW greedy placement
    payloads.txt, decodes.csv                    our extras (naive = tag scan like the authors', smart = tries every tag)

Usage:  python3 ap_step4d_generate_datasets.py                (baseline: retention 0, key 0, fragment sizes 1-5, 10 datasets)
        python3 ap_step4d_generate_datasets.py 1 2 3 ...      (explicit lists: sizes retentions keys datasets, comma separated)
"""
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

OUT = ROOT / "data/datasets"
M = sc.load_matrix(ROOT / "matrices/PAM10")
enc.START_TAG, enc.END_TAG = "TCATG", "GTTGG"  # the authors' experiment tags
NAMES = {"clean": "non_trojan_dataset.txt",
         "random": "random_trojan_insertion_dataset.txt",
         "best": "nw_best_greedy_trojan_insertion_dataset.txt",
         "worst": "nw_worst_greedy_trojan_insertion_dataset.txt"}


def seed_data(n=1000, seed=0):
    """Hosts = first 1000 reads; payloads = 1000 fake addresses like 'xy42ab' (authors' seed_data_generator + make_shorter_payloads)."""
    hosts = sw.load_reads()[:n]
    rng = np.random.default_rng(seed)
    return hosts, [sw.authors_payload(rng) for _ in range(n)]


def recovers(decoder, seq, key, retention, payload):
    """True if the decoder returns the payload. A crash (e.g. a natural tag swallowed a fragment and left nothing to decode)
    counts as a failure: a real naive Trojan would not recover the payload either."""
    try:
        return decoder(seq, key, retention) == payload
    except (ValueError, KeyError, IndexError):
        return False


def generate(size, retention, key, n_datasets=10, seed=0):
    """Write all datasets for one (fragment size, retention, key) setting. Returns decode success counts."""
    hosts, payloads = seed_data()
    rng = np.random.default_rng([seed, size, retention, key])
    folder = OUT / f"fragment_len_{size}/retention_pos_{retention}/encryption_key_{key}"
    ok = {"random": 0, "best": 0, "worst": 0}
    ok_smart = {"random": 0, "best": 0, "worst": 0}
    total = 0
    for d in range(n_datasets):
        path = folder / f"dataset_{d}"
        path.mkdir(parents=True, exist_ok=True)
        rows = {k: [] for k in NAMES}
        meta = []
        for k, h in enumerate(rng.permutation(len(hosts))):  # authors: random.sample(hosts, 1000), zip with payloads
            host, payload = hosts[h], payloads[k]
            frags = enc.make_fragments(payload, size, key, retention)
            seqs = {"clean": host,
                    "random": gr.place_random(host, frags, rng)[0],
                    "best": gr.place_greedy(host, frags, M, best=True)[0],
                    "worst": gr.place_greedy(host, frags, M, best=False)[0]}
            line = []
            for name in ("random", "best", "worst"):
                good = recovers(enc.read_payload, seqs[name], key, retention, payload)  # naive tag-scanning Trojan
                smart = recovers(enc.read_payload_robust, seqs[name], key, retention, payload)  # smarter Trojan
                ok[name] += good
                ok_smart[name] += smart
                line += [int(good), int(smart)]
            total += 1
            for name, seq in seqs.items():
                rows[name].append(seq)
            meta.append(f"{k},{h},{payload}," + ",".join(map(str, line)))
        for name, fn in NAMES.items():
            (path / fn).write_text("\n".join(rows[name]) + "\n")
        (path / "payloads.txt").write_text("\n".join(m.split(",")[2] for m in meta) + "\n")
        (path / "decodes.csv").write_text("line,host_id,payload,naive_random,smart_random,naive_best,smart_best,naive_worst,smart_worst\n" + "\n".join(meta) + "\n")
    return {k: v / total for k, v in ok.items()}, {k: v / total for k, v in ok_smart.items()}


if __name__ == "__main__":
    a = [[int(x) for x in arg.split(",")] for arg in sys.argv[1:]]
    sizes, rets, keys, dsets = (a + [[1, 2, 3, 4, 5], [0], [0], [10]][len(a):])[:4] if a else ([1, 2, 3, 4, 5], [0], [0], [10])
    for size in sizes:
        for ret in rets:
            for key in keys:
                naive, smart = generate(size, ret, key, n_datasets=dsets[0] if isinstance(dsets, list) else dsets)
                print(f"fragment {size}, retention {ret}, key {key}: payload recovered (naive scan | smart)  "
                      + "  ".join(f"{n}: {100 * naive[n]:.0f}% | {100 * smart[n]:.0f}%" for n in ("random", "best", "worst")), flush=True)
