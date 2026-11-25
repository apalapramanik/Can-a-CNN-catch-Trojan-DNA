"""Step 4b: place SEVERAL fragments into one host read, one after another (paper Eqs. 5-6).

WHO DID WHAT
  Authors: the greedy strategy and their code, get_nw_based_insertions_greedy in authors_repo/trojan_attack/nw_insertion_new.py,
    plus the random placement in generate_dataset_new.py (RandInsertion.get_random_insertion).
  Us: the same strategies rewritten with numpy (no re-scoring of the whole 1000-letter read for every try: a
    fragment's score change only depends on the letters it overwrites). ap_step4b_test.py checks that we place
    fragments exactly where the authors' code does.

How greedy works: take the fragments in order. For each one, try every start position that does not overlap an
already-placed fragment, keep the best (MaxNW) or worst (MinNW) one, lock those positions, move on.
Because earlier fragments are locked, a new fragment's score change depends only on the ORIGINAL host letters it covers.
"""
import numpy as np

import ap_step4_score as sc

LETTERS = "ACGTN"


def loss_table(matrix):
    """loss[c][h] = how much the score drops when payload letter c overwrites host letter h
    (= matrix[h][h] - matrix[h][c]). Rows: payload letter, columns: host letter, both indexed by LETTERS."""
    return np.array([[matrix[(h, h)] - matrix[(h, c)] for h in LETTERS] for c in LETTERS])


def fragment_losses(host_idx, fragment, table):
    """Score drop for putting `fragment` at every start position i (0 .. len(host)-len(fragment)). '*' slots cost nothing."""
    n = len(host_idx) - len(fragment) + 1
    total = np.zeros(n, dtype=np.int64)
    for j, c in enumerate(fragment):
        if c != "*":
            total += table[LETTERS.index(c)][host_idx[j:j + n]]
    return total


def place_greedy(host, fragments, matrix, best=True):
    """MaxNW (best=True) or MinNW (best=False) greedy placement. Returns (injected read, start position of each fragment).
    Ties go to the earliest position, like the authors' code (they only switch on a strictly better score)."""
    host_idx = np.array([LETTERS.index(c) for c in host])
    table = loss_table(matrix)
    locked = np.zeros(len(host), dtype=int)
    out = list(host)
    starts = []
    for frag in fragments:
        loss = fragment_losses(host_idx, frag, table)
        # a start i is blocked if any position i .. i+len-1 is already locked
        csum = np.concatenate([[0], np.cumsum(locked)])
        blocked = (csum[len(frag):len(frag) + len(loss)] - csum[:len(loss)]) > 0
        if blocked.all():
            raise ValueError("no room left for a fragment")
        pick = np.where(blocked, np.inf if best else -np.inf, loss.astype(float))
        i = int(np.argmin(pick) if best else np.argmax(pick))
        for j, c in enumerate(frag):
            if c != "*":
                out[i + j] = c
        locked[i:i + len(frag)] = 1
        starts.append(i)
    return "".join(out), starts


def place_random(host, fragments, rng):
    """The authors' random strategy: fragments keep their order, each starts somewhere to the right of the previous one."""
    out = list(host)
    remaining = sum(len(f) for f in fragments)
    start_from, starts = 0, []
    for frag in fragments:
        end_to = len(host) - remaining
        i = int(rng.integers(start_from, end_to))
        for j, c in enumerate(frag):
            if c != "*":
                out[i + j] = c
        starts.append(i)
        start_from = i + len(frag)
        remaining -= len(frag)
    return "".join(out), starts


# --- fake payloads like the authors' (First.Last:port, full name at most 10 characters incl. the space) ---
_FIRST = ["Ann", "Bob", "Carl", "Dan", "Eve", "Fay", "Gus", "Hal", "Ian", "Jo", "Kim", "Lee", "Max", "Ned", "Pam", "Ray", "Sue", "Tom", "Una", "Vic"]
_LAST = ["Cole", "Reed", "Hall", "King", "Ross", "Wood", "Lane", "Park", "Webb", "Young", "Fox", "Hill", "Kerr", "Moss", "Orr", "Pratt", "Shaw", "Todd", "Vale", "West"]


def fake_payload(rng):
    """Random 'First.Last:port' with len(First)+len(Last) <= 9 (authors' rule: full name <= 10 chars) and port 1..9999."""
    while True:
        f, l = _FIRST[rng.integers(len(_FIRST))], _LAST[rng.integers(len(_LAST))]
        if len(f) + len(l) <= 9:
            return f"{f}.{l}:{int(rng.integers(1, 10000))}"
