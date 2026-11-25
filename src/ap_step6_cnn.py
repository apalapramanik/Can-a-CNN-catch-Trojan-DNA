"""Step 6: train a small CNN to tell clean reads from injected reads.

WHO DID WHAT
  Authors: trojan_cnn.py (Keras). Their CODE uses 2 conv layers (32 filters, kernel 5), pool 2, dense 16, dropout, L2, SGD 3000 epochs,
    and a random line-level 75/25 split of one dataset. The PAPER text describes 1 conv layer, 16 filters, kernel 16, max-pool 4.
  Us: PyTorch re-implementation of BOTH variants (arch="paper" / arch="code"), Adam optimiser and few epochs (our choice, for speed),
    and two ways to split train/test: "host" (a host is wholly in train or test, as in step 5) or "line" (random lines, like the authors).
    Comparing the two shows how much of the accuracy could come from memorising hosts.

One-hot input: each letter -> 5 numbers (A C T G N, same column order as the authors); a read is a 5 x 1000 grid.

Usage (venv python): ~/venvs/ap_trojan/bin/python -I ap_step6_cnn.py [size] [retention] [scenario] [arch] [split] [epochs]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ap_step5_kmer_check as s5  # reuse load() and host_split()

ORDER = "ACTGN"
LUT = np.full(256, 4, dtype=np.int64)  # unknown letters -> N slot
for i, ch in enumerate(ORDER):
    LUT[ord(ch)] = i


def one_hot(reads):
    """list of 1000-letter strings -> float32 array (n, 5, 1000)."""
    idx = LUT[np.frombuffer("".join(reads).encode(), dtype=np.uint8)].reshape(len(reads), -1)
    out = np.zeros((len(reads), 5, idx.shape[1]), dtype=np.float32)
    np.put_along_axis(out, idx[:, None, :], 1.0, axis=1)
    return out


def build_model(arch, length=1000):
    if arch == "paper":  # 1 conv layer, 16 filters, kernel 16, max-pool 4
        conv = nn.Sequential(nn.Conv1d(5, 16, 16), nn.ReLU(), nn.MaxPool1d(4), nn.Flatten())
        n = 16 * ((length - 15) // 4)
        return nn.Sequential(conv, nn.Linear(n, 1))
    if arch in ("gmp", "wide"):  # OUR variants: global max-pool = "does this pattern appear ANYWHERE in the read?"
        f, k = (16, 16) if arch == "gmp" else (64, 24)
        return nn.Sequential(nn.Conv1d(5, f, k), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(f, 1))
    if arch == "code":  # authors' repo: 2 x (conv 32, k5, same), pool 2, dense 16
        conv = nn.Sequential(
            nn.Conv1d(5, 32, 5, padding=2), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.Dropout(0.2),
            nn.MaxPool1d(2), nn.Flatten())
        return nn.Sequential(conv, nn.Linear(32 * (length // 2), 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))
    raise ValueError(arch)


def line_split(n, seed=42, test_fraction=0.25):
    test = np.zeros(n, dtype=bool)
    test[np.random.default_rng(seed).choice(n, int(n * test_fraction), replace=False)] = True
    return test


def run(size=1, retention=0, scenario="best", arch="paper", split="host", epochs=20, key=0, n_datasets=4, seed=0, verbose=True, return_scores=False, save_dir=None):
    torch.manual_seed(seed)
    reads, y, hosts = s5.load(size, retention, key, scenario, n_datasets)
    X = torch.from_numpy(one_hot(reads))
    yt = torch.from_numpy(y.astype(np.float32))
    test = s5.host_split(hosts, seed) if split == "host" else line_split(len(y))
    tr, te = torch.from_numpy(~test), torch.from_numpy(test)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr, ytr, Xte, yte = X[tr].to(dev), yt[tr].to(dev), X[te].to(dev), yt[te]
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 100):
            b = perm[i:i + 100]
            opt.zero_grad()
            loss_fn(model(Xtr[b]).squeeze(1), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(Xte[i:i + 500]).squeeze(1).cpu() for i in range(0, len(Xte), 500)])
        acc = ((p > 0).float() == yte).float().mean().item()
        auc = roc_auc_score(yte.numpy(), p.numpy())
        hist.append((acc, auc))
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}  test acc {acc * 100:5.1f}%  AUC {auc:.3f}  ({time.time() - t0:.0f}s)")
    if save_dir is not None:  # keep the trained weights (state_dict); rebuild with build_model(arch) then load_state_dict
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"{arch}_size{size}_ret{retention}_{scenario}_key{key}_{split}_seed{seed}_ep{epochs}.pt")
    return (hist, yte.numpy(), p.numpy()) if return_scores else hist


if __name__ == "__main__":
    a = sys.argv[1:]
    size, ret = int(a[0]) if a else 1, int(a[1]) if len(a) > 1 else 0
    scen = a[2] if len(a) > 2 else "best"
    arch, split = (a[3] if len(a) > 3 else "paper"), (a[4] if len(a) > 4 else "host")
    ep = int(a[5]) if len(a) > 5 else 20
    print(f"fragment {size}, retention {ret}, scenario {scen}, arch {arch}, split {split}, {ep} epochs")
    run(size, ret, scen, arch, split, ep)
