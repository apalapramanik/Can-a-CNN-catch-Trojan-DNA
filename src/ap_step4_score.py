"""Step 4a: the similarity score between a host read and an injected read, and the best/worst/random injection spot.

WHO DID WHAT
  Authors: the idea (pick the injection spot that keeps the injected read most similar to the host, "MaxNW"),
    and their code in authors_repo/trojan_attack/nw_insertion.py, which calls the library function
    nwalign3.score_alignment with the PAM250 matrix.
  Us: a from-scratch scorer that mimics that library function (we checked it against the 2 examples in the
    nwalign3 docs), and the search over positions. Our search uses one block of payload; the authors'
    multi-fragment 'greedy' placement (paper Eqs. 5-6) is the next step.

FINDING: nwalign3.score_alignment scores an alignment that ALREADY EXISTS ("the first 2 args are from an
  alignment and must have the same length"). The authors give it two same-length strings with no '-' gaps, so their
  "NW score" is just the sum, position by position, of matrix[host letter][injected letter]. No gap search happens.
  The paper says PAM10 with gap penalties 15.79/1.29; the authors' code uses PAM250 (and gap values that never
  matter without gaps). We follow the code, since that is what produced their results.
"""
import numpy as np


def load_matrix(path):
    """Read a PAM-style matrix file (like authors_repo/trojan_attack/PAM250) into {(x, y): score}.
    Lines starting with '#' are comments; the first other line lists the letters; each row starts with its letter."""
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    letters = rows[0]
    return {(r[0], letters[k]): int(v) for r in rows[1:] for k, v in enumerate(r[1:])}


def score_alignment(a, b, matrix, gap_open=-5, gap_extend=-2):
    """Score two aligned strings of equal length ('-' = gap). Mimics nwalign3.score_alignment:
    a letter pair scores matrix[x][y]; the first gap column costs gap_open and each following gap column in a
    row costs gap_extend (a gap in either sequence continues the same run). Checked on the library's doc examples."""
    assert len(a) == len(b)
    score, in_gap = 0, False
    for x, y in zip(a, b):
        if x == "-" or y == "-":
            score += gap_extend if in_gap else gap_open
            in_gap = True
        else:
            score += matrix[(x, y)]
            in_gap = False
    return score


def ungapped_score(host, inj, matrix):
    """The authors' effective 'NW score': sum of matrix[host letter][injected letter] over all positions."""
    return sum(matrix[(h, x)] for h, x in zip(host, inj))


def scores_for_all_positions(host, payload_masked, matrix):
    """Score of the injected read for every possible start position i (paper Eq. 2: i = 0 .. len(host)-len(payload)).
    Equivalent to ungapped_score(host, inject(host, payload, i)) for each i, but only the overwritten letters
    are recomputed, so it is fast. '*' slots keep the host letter and change nothing."""
    base = ungapped_score(host, host, matrix)
    slots = [(j, c) for j, c in enumerate(payload_masked) if c != "*"]
    out = np.empty(len(host) - len(payload_masked) + 1)
    for i in range(len(out)):
        loss = 0
        for j, c in slots:
            h = host[i + j]
            loss += matrix[(h, h)] - matrix[(h, c)]
        out[i] = base - loss
    return out


def pick_positions(scores, rng):
    """The three attacker strategies of paper Fig. 5b: MaxNW (best), MinNW (worst), Random."""
    return {"MaxNW": int(np.argmax(scores)), "MinNW": int(np.argmin(scores)), "Random": int(rng.integers(len(scores)))}
