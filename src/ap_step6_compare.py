"""Step 6b: does longer training / global max-pool / wider filters help the CNN on the hard setting? 5 seeds each, host split.
Usage (venv python): python -I ap_step6_compare.py [size] [retention]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import ap_step6_cnn as c

size = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ret = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print(f"fragment {size}, retention {ret}, MaxNW, host split (host split seed varies with run seed)")
for arch, ep in (("paper", 40), ("paper", 150), ("gmp", 150), ("wide", 150)):
    accs, aucs = [], []
    for seed in range(5):
        h = c.run(size, ret, "best", arch, "host", ep, seed=seed, verbose=False)
        accs.append(h[-1][0]); aucs.append(h[-1][1])
    print(f"{arch:6s} {ep:3d} epochs: acc {np.mean(accs)*100:5.1f}% +- {np.std(accs)*100:.1f}   AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}", flush=True)
