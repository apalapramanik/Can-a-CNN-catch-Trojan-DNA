"""Step 18: PyTorch copy of the CNN of Fig. 7b of the paper (2 conv layers, 16 filters, kernel 8, max pool 4), trained with the authors' settings.

WHY
  The Original CNN (authors' Keras script) and the Reimplemented CNN used different architectures and training setups, so they were
  not comparable. This script uses the Fig. 7b architecture and the authors' training settings, so the only change left against the
  Keras run (step 9, run with ARCH=fig7b) is the framework.

ARCHITECTURE (paper, Fig. 7b: filter 16, kernel 8, max pool 4, 2 hidden layers; dense 16 and dropouts as in the authors' code)
  Conv1D(16, k8, same, he_uniform, relu, L2 0.001) - Dropout 0.2 - Conv1D(16, k8, same, he_uniform, relu, L2 0.001) - Dropout 0.2
  - MaxPool 4 - Flatten - Dense(16, he_uniform, relu, L2 0.001) - Dropout 0.1 - Dense(2, softmax)
TRAINING (authors' code and paper text)
  SGD lr 0.001, momentum 0.9, Keras decay lr/(1 + decay*step) with decay = lr/epochs, batch 100, 3000 epochs,
  binary cross entropy on the 2 softmax outputs, validation_split 0.10 (last 10 percent of the training reads).
DATA
  one dataset (clean file + trojan file), lines cut or padded to 1000, one-hot over A C T G N, 1000 clean + 1000 trojan reads.
  split="line": sklearn train_test_split(test_size=0.25, random_state=42), the authors' read level split.
  split="host": both copies of a host (same line number in the two files) go to the same side. Used later.

Usage (venv python): DEVICE=mps ~/venvs/ap_trojan/bin/python -I ap_step18_same_as_original.py size retention [epochs=3000] [split=line] [seed=0] [datasets=1]
  datasets=10 pools datasets 0..9 as the paper does (10,000 clean + 10,000 trojan reads, 75/25 random read level split per class).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEN, ORDER = 1000, "ACTGN"
FILTERS, KERNEL, POOL = 16, 8, 4  # Fig. 7b
LUT = np.full(256, 4, dtype=np.int64)
for i, ch in enumerate(ORDER):
    LUT[ord(ch)] = i


def pad(line):  # same as the authors' padding(): repeat the start of the line
    while len(line) < LEN:
        line = line + line[:LEN - len(line)]
    return line


def read_file(path):
    return [(l.strip()[:LEN] if len(l.strip()) >= LEN else pad(l.strip())) for l in open(path)]


def one_hot(reads):
    idx = LUT[np.frombuffer("".join(reads).encode(), dtype=np.uint8)].reshape(len(reads), -1)
    out = np.zeros((len(reads), 5, LEN), dtype=np.float32)
    np.put_along_axis(out, idx[:, None, :], 1.0, axis=1)
    return out


def same_pad(k):  # Keras padding="same" for stride 1: total k-1, extra on the right when k is even
    return (k - 1) // 2, (k - 1) - (k - 1) // 2


class Fig7bCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(5, FILTERS, KERNEL), nn.Conv1d(FILTERS, FILTERS, KERNEL)
        self.d1, self.d2 = nn.Linear(FILTERS * (LEN // POOL), 16), nn.Linear(16, 2)
        for m in (self.c1, self.c2, self.d1):  # he_uniform, zero bias
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.d2.weight); nn.init.zeros_(self.d2.bias)  # Keras default for the last layer

    def forward(self, x):
        x = F.dropout(F.relu(self.c1(F.pad(x, same_pad(KERNEL)))), 0.2, self.training)
        x = F.dropout(F.relu(self.c2(F.pad(x, same_pad(KERNEL)))), 0.2, self.training)
        x = F.max_pool1d(x, POOL).flatten(1)
        x = F.dropout(F.relu(self.d1(x)), 0.1, self.training)
        return self.d2(x)

    def l2(self):  # Keras l2(0.001) on the kernels of the two conv layers and the 16 unit dense layer
        return 0.001 * sum((m.weight ** 2).sum() for m in (self.c1, self.c2, self.d1))


class WideCNN(nn.Module):
    """Our Wide CNN: 1 conv layer of 64 filters, kernel 24, ReLU, global max pool, Dense(2, softmax). Same loss and training as Fig7bCNN."""

    def __init__(self):
        super().__init__()
        self.c1, self.d2 = nn.Conv1d(5, 64, 24), nn.Linear(64, 2)
        nn.init.kaiming_uniform_(self.c1.weight, nonlinearity="relu"); nn.init.zeros_(self.c1.bias)
        nn.init.xavier_uniform_(self.d2.weight); nn.init.zeros_(self.d2.bias)

    def forward(self, x):
        return self.d2(F.relu(self.c1(x)).amax(2))

    def l2(self):  # L2 0.001 on the conv kernel, as for the conv layers of the authors' model
        return 0.001 * (self.c1.weight ** 2).sum()


def run(size, ret, epochs=3000, split="line", seed=0, scen="best", ds=0, log_every=100, nds=1):
    torch.manual_seed(seed); np.random.seed(seed)
    root = ROOT / f"data/datasets/fragment_len_{size}/retention_pos_{ret}/encryption_key_0"
    dsl = [ds] if nds == 1 else list(range(nds))  # nds > 1 pools datasets 0..nds-1, as the paper does (10 clean + 10 trojan datasets)
    files = {"best": "nw_best_greedy_trojan_insertion_dataset.txt", "random": "random_trojan_insertion_dataset.txt",
             "worst": "nw_worst_greedy_trojan_insertion_dataset.txt"}
    clean, troj, hid = [], [], []
    for d in dsl:
        base = root / f"dataset_{d}"
        clean += read_file(base / "non_trojan_dataset.txt"); troj += read_file(base / files[scen])
        hid += [int(l.split(",")[1]) for l in (base / "decodes.csv").read_text().split("\n")[1:-1]]
    n = len(clean)
    X = torch.from_numpy(one_hot(clean + troj))
    y = torch.from_numpy(np.array([0] * n + [1] * len(troj)))
    idx = np.arange(len(y))
    if split == "line":
        tr, te = train_test_split(idx, test_size=0.25, random_state=42, stratify=y.numpy() if nds > 1 else None)
    else:  # host level: decodes.csv gives the host id of each line, the same for the clean file and the trojan file
        assert len(hid) == n, (len(hid), n)
        host_of = np.array(hid + hid)
        ids = np.unique(host_of); np.random.default_rng(42).shuffle(ids)
        te_h = set(ids[: int(0.25 * len(ids))].tolist())
        is_te = np.array([h in te_h for h in host_of])
        te, tr = idx[is_te], idx[~is_te]
        tr = np.random.default_rng(42).permutation(tr)  # shuffle like train_test_split, so the last 10 percent (validation) mixes both classes
    nval = int(0.10 * len(tr))  # Keras validation_split takes the LAST 10 percent
    fit_i, val_i = tr[: len(tr) - nval], tr[len(tr) - nval:]
    dev = os.environ.get("DEVICE", "cpu")
    model = (WideCNN() if os.environ.get("ARCH") == "wide" else Fig7bCNN()).to(dev)
    print("trainable parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    lr0, decay = 0.001, 0.001 / epochs
    opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9)
    X, y = X.to(dev), y.to(dev)
    Xf, yf = X[fit_i], F.one_hot(y[fit_i], 2).float()
    step, t0 = 0, time.time()

    def acc(ii):
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(X[ii[k:k + 500]]) for k in range(0, len(ii), 500)])
        return (p.argmax(1) == y[ii]).float().mean().item(), p.argmax(1).cpu().numpy()

    patience = int(os.environ.get("PATIENCE", "0"))  # early stopping on validation loss; 0 = off (train all epochs)
    min_ep = int(os.environ.get("MIN_EPOCHS", "0"))  # stopping cannot trigger before this epoch: the loss can sit at 0.693 for 300 epochs before learning starts
    best_v, best_ep, best_sd = float("inf"), 0, None
    yv = F.one_hot(y[val_i], 2).float()

    def val_loss():
        model.eval()
        with torch.no_grad():
            p = torch.cat([F.softmax(model(X[val_i[k:k + 500]]), 1) for k in range(0, len(val_i), 500)]).clamp(1e-7, 1 - 1e-7)
        return F.binary_cross_entropy(p, yv).item()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xf), device=dev)
        for k in range(0, len(perm), 100):
            b = perm[k:k + 100]
            for g in opt.param_groups:
                g["lr"] = lr0 / (1 + decay * step)
            opt.zero_grad()
            pr = F.softmax(model(Xf[b]), 1).clamp(1e-7, 1 - 1e-7)
            (F.binary_cross_entropy(pr, yf[b]) + model.l2()).backward()
            opt.step(); step += 1
        if patience:
            v = val_loss()
            if v < best_v:
                best_v, best_ep, best_sd = v, ep + 1, {k: t.clone() for k, t in model.state_dict().items()}
            elif ep + 1 - best_ep >= patience and ep + 1 >= min_ep:
                print(f"early stop at epoch {ep + 1}, best epoch {best_ep}, best validation loss {best_v:.4f}", flush=True)
                break
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"epoch {ep + 1:4d}  train {acc(fit_i)[0] * 100:5.1f}  val {acc(val_i)[0] * 100:5.1f}  test {acc(te)[0] * 100:5.1f}  ({time.time() - t0:.0f}s)", flush=True)
    if patience and best_sd is not None:
        model.load_state_dict(best_sd)  # keep the weights of the best validation epoch
    a, pred = acc(te)
    yt = y[te].cpu().numpy()
    cm = np.array([[((yt == i) & (pred == j)).sum() for j in (0, 1)] for i in (0, 1)], dtype=float)
    cm = cm / cm.sum(1, keepdims=True)
    out = ROOT / "results/step18_same_as_original"
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"pool{nds}" if nds > 1 else f"ds{ds}") + (f"_es{patience}m{min_ep}" if patience else "") + ("_wide" if os.environ.get("ARCH") == "wide" else "")
    f = out / f"size{size}_ret{ret}_{split}_{tag}_ep{epochs}_seed{seed}.txt"
    model.eval()
    with torch.no_grad():  # keep test scores (logit of Trojan minus logit of clean) for ROC and PR curves later
        sc = torch.cat([model(X[te[k:k + 500]]) for k in range(0, len(te), 500)]).cpu().numpy()
    np.savez(f.with_suffix(".npz"), y=yt, score=sc[:, 1] - sc[:, 0])
    f.write_text("TN, FN, FP, TP, accuracy\n" + f"{cm[0][0]}, {cm[0][1]}, {cm[1][0]}, {cm[1][1]}, {a * 100}\n")
    print(f"stopped after epoch {ep + 1}, best epoch {best_ep}" if patience else "")
    print(f"done in {time.time() - t0:.0f}s, test accuracy {a * 100:.1f}%, saved {f}")
    return a


if __name__ == "__main__":
    a = sys.argv[1:]
    run(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 3000, a[3] if len(a) > 3 else "line", int(a[4]) if len(a) > 4 else 0,
        nds=int(a[5]) if len(a) > 5 else 1)
