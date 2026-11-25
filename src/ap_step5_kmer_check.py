"""Step 5: can a plain k-mer frequency model tell clean reads from injected reads (no neural network)?

WHO DID WHAT
  Authors: the data and the CNN detector (trojan_cnn.py); they never test a simpler baseline.
  Us: this baseline, as proposed in the research-directions doc (day-2 experiment). A k-mer is a DNA word of k letters; a read's
    k-mer frequency vector says how often each word occurs. We train a logistic regression on those vectors.
  Design choices (ours): train and test are split BY HOST (a host seen in training never appears in testing, otherwise the model
    could memorise hosts, since every injected read has a clean twin from the same host). Positives = injected reads, negatives = clean.

Usage: python3 ap_step5_kmer_check.py            (about a few minutes)
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ap_step4d_generate_datasets as gen

FILES = gen.NAMES
SCEN = {"MaxNW": "best", "Random": "random", "MinNW": "worst"}
N_DATASETS = 4  # datasets 0..3 pooled (each: 1000 hosts, 1000 clean + 1000 injected per scenario)


def load(size, retention=0, key=0, scenario="best", n_datasets=N_DATASETS):
    """Return reads, labels (0 clean / 1 injected), host ids, for datasets 0..n-1 pooled."""
    base = ROOT / f"data/datasets/fragment_len_{size}/retention_pos_{retention}/encryption_key_{key}"
    reads, labels, hosts = [], [], []
    for d in range(n_datasets):
        ds = base / f"dataset_{d}"
        hid = [int(l.split(",")[1]) for l in (ds / "decodes.csv").read_text().split("\n")[1:-1]]
        for fn, lab in ((FILES["clean"], 0), (FILES[scenario], 1)):
            r = (ds / fn).read_text().split("\n")[:-1]
            reads += r; labels += [lab] * len(r); hosts += hid
    return reads, np.array(labels), np.array(hosts)


def kmer_features(reads, k):
    """Row = one read, column = one k-mer, value = how often it occurs / number of positions (a frequency)."""
    cv = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False, dtype=np.float32)
    X = cv.fit_transform(reads)
    return X / (len(reads[0]) - k + 1), cv.get_feature_names_out()


def host_split(hosts, seed=0, test_fraction=0.25):
    """Boolean mask of TEST rows: 25% of the host ids, chosen once, so each host is entirely train or entirely test."""
    ids = np.unique(hosts)
    test_ids = set(np.random.default_rng(seed).choice(ids, int(len(ids) * test_fraction), replace=False))
    return np.array([h in test_ids for h in hosts])


def evaluate(size, k, scenario, retention=0, key=0):
    reads, y, hosts = load(size, retention, key, scenario)
    X, names = kmer_features(reads, k)
    test = host_split(hosts)
    sc = MaxAbsScaler().fit(X[~test])
    clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(X[~test]), y[~test])
    p = clf.predict_proba(sc.transform(X[test]))[:, 1]
    return float(((p > 0.5) == y[test]).mean()), float(roc_auc_score(y[test], p)), clf, names


if __name__ == "__main__":
    t0 = time.time()
    print("0) duplicates check: distinct clean hosts across the 4 pooled datasets =",
          len(set(load(1)[2])), "(each dataset reuses the same 1000 hosts, reshuffled)\n")

    print("1) test accuracy / AUC of logistic regression on k-mer frequencies, fragment size 1..5, retention 0, no encryption")
    print("   (50% accuracy = chance; hosts in the test set were never seen in training)")
    res = {}
    for k in (3, 4, 5, 6):
        print(f"\n   k = {k}")
        print("   fragment size |      MaxNW        |      Random       |      MinNW        (accuracy / AUC)")
        for size in range(1, 6):
            row = f"        {size}        |"
            for name, sce in SCEN.items():
                acc, auc, _, _ = evaluate(size, k, sce)
                res[(k, size, name)] = (acc, auc)
                row += f"  {100 * acc:5.1f}% / {auc:.3f}  |"
            print("  ", row, flush=True)
    print(f"\n   ({time.time() - t0:.0f}s so far)")

    # what does the model look at? top k-mers for the hardest case
    acc, auc, clf, names = evaluate(5, 6, "best")
    top = np.argsort(clf.coef_[0])[::-1][:10]
    print("\n2) the 10 6-mers the model weighs most for 'injected' (fragment size 5, MaxNW):", ", ".join(names[top]))
    print("   (tags are TCATG and GTTGG: do their 6-mers appear above?)")

    import json
    (ROOT / "results/step5_kmer.json").write_text(json.dumps({f"{k}|{s}|{n}": v for (k, s, n), v in res.items()}))
    print(f"\ndone in {time.time() - t0:.0f}s")
