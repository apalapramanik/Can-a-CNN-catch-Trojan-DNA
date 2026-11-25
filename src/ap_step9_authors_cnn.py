"""Step 9b: run the AUTHORS' CNN (trojan_cnn.py main()) on our generated data.

WHO DID WHAT
  Authors: trojan_cnn.py (Keras 2.3: 2 conv layers x32 filters kernel 5, dropout, L2, dense 16, SGD lr 0.001 momentum 0.9, 3000 epochs,
    batch 100, validation_split 0.10, random LINE split 75/25 of ONE dataset with random_state 42).
  Us: this runner. Their file is read from authors_repo and never edited. It was written for Keras 2.3, so we apply 4 compatibility
  rewrites in memory (printed below) so it runs on today's Keras 3, and call their main() directly with our file paths:
    (a) SGD(lr=.., decay=..)  -> SGD(learning_rate=InverseTimeDecay(lr, 1, decay))  (Keras 2's decay rule: lr / (1 + decay * step))
    (b) fit(nb_epoch=..)      -> fit(epochs=..)   (renamed)
    (c) fit(.., verbose=0)    (silence 3000 lines of output)
    (d) the epoch count can be overridden by us (default: their 3000)
    (e) model.save(...) added before their test step (they never save), to ../models/
  We bypass their command-line parser: it has bugs (retention_pos and encryption_key are taken from --dataset_num; retention 5 is rejected
  because range(0, 5)), so it cannot select retention>0 correctly.

Usage (TensorFlow env): ~/venvs/ap_authors/bin/python -I ap_step9_authors_cnn.py size retention [epochs=3000] [dataset=0] [scenario=best]
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

size, ret = int(sys.argv[1]), int(sys.argv[2])
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
ds = int(sys.argv[4]) if len(sys.argv) > 4 else 0
scen = sys.argv[5] if len(sys.argv) > 5 else "best"
FILES = {"best": "nw_best_greedy_trojan_insertion_dataset.txt", "random": "random_trojan_insertion_dataset.txt",
         "worst": "nw_worst_greedy_trojan_insertion_dataset.txt"}

src = (ROOT / "authors_repo/trojan_attack/trojan_cnn.py").read_text()
patches = [
    ("sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)",
     "sgd = SGD(learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(lrate, 1, decay), momentum=0.9, nesterov=False)"),
    ("nb_epoch=epochs,", "epochs=EPOCHS,"),
    ("validation_split=0.10)", "validation_split=0.10, verbose=0)"),
    ("    epochs = 3000\n", "    epochs = EPOCHS\n"),
    ("    # test results\n", "    model.save(MODEL_PATH)  # added by us: the authors never save the model\n    # test results\n"),
]
for old, new in patches:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
print("compatibility rewrites applied:", len(patches))
# Added in step 18: ARCH=fig7b swaps the architecture for the one in Fig. 7b of the paper (16 filters, kernel 8, max pool 4, 2 conv layers)
TAG = ""
if os.environ.get("ARCH") == "fig7b":
    for old, new in (("filters=32", "filters=16"), ("kernel_size=5", "kernel_size=8"), ("pool_size=2", "pool_size=4")):
        assert src.count(old) >= 1, old
        src = src.replace(old, new)
    TAG = "_fig7b"
    print("architecture changed to Fig. 7b: 16 filters, kernel 8, max pool 4")

import tensorflow as tf
(ROOT / "models").mkdir(exist_ok=True)
ns = {"__name__": "authors_trojan_cnn", "EPOCHS": epochs, "tf": tf,
      "MODEL_PATH": str(ROOT / f"models/authors_cnn{TAG}_size{size}_ret{ret}_{scen}_ds{ds}_ep{epochs}.keras")}
exec(compile(src, "trojan_cnn.py(patched)", "exec"), ns)

base = ROOT / f"data/datasets/fragment_len_{size}/retention_pos_{ret}/encryption_key_0/dataset_{ds}/"
out = ROOT / ("results/step9_authors_cnn_fig7b" if TAG else "results/step9_authors_cnn")
out.mkdir(parents=True, exist_ok=True)
res = out / f"size{size}_ret{ret}_{scen}_ds{ds}_ep{epochs}.txt"
t0 = time.time()
ns["main"](str(base / "non_trojan_dataset.txt"), str(base / FILES[scen]), str(res), str(out / "logs"))
print(f"done in {time.time() - t0:.0f}s; result file {res}")
print(res.read_text())
