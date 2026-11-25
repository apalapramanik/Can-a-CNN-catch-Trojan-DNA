# Step 10: how the consensus/majority-vote simulation works

This explains, in plain terms, the code in `src/ap_step10_consensus_tradeoff.py`,
`src/ap_step10b_fixed_count_sweep.py`, and `src/ap_step10c_infection_probability.py`, how the
plots are built, and the math behind the "opposite trends" result (`figures/step10c_infection_probability.png`).

## 1. The scenario, in one paragraph

A DNA region gets sequenced not once but **N times** (N "copies"). Some of those copies might carry an
attacker's hidden edit; the rest are the clean, true sequence. A piece of software lines up all N copies and,
at every position, outputs whatever letter the **majority** of copies agree on. We're simulating that
majority vote on a computer, with a controllable number of copies (N), a controllable amount of ordinary
random sequencing error, and a controllable amount of attacker interference, so we can see when the vote
outputs the true sequence and when it outputs the attacker's edit instead.

## 2. The building blocks (`ap_step10_consensus_tradeoff.py`)

```python
def noisy(chars, p_err, rng):
    out = chars.copy()
    flip = rng.random(len(chars)) < p_err        # a coin flip at every position
    out[flip] = rng.choice(BASES, flip.sum())     # those positions get a random new letter
    return out
```
**In simple terms:** this is one "copy" coming out of the sequencing machine with mistakes in it. For every
letter, we flip a coin that lands "error" with probability `p_err`. Wherever it lands error, we replace that
letter with a random one of A/C/G/T. This is our stand-in for real sequencing error.

```python
def majority_consensus(copies):
    counts = np.stack([(copies == b).sum(axis=0) for b in BASES])  # how many copies say A? C? G? T? at each position
    return BASES[np.argmax(counts, axis=0)]                        # pick the letter with the most votes, per position
```
**In simple terms:** imagine N people shouting out a letter for the same position, one letter each. This
counts the shouts for A, the shouts for C, the shouts for G, the shouts for T, and declares whichever got the
most shouts the winner — separately, for every position in the read, all at once (that's what the "stack" and
"argmax" are doing — no explicit for-loop over positions, for speed).

```python
def run_trial_count(host_region, inj_region, N, n_mal, p_err, rng):
    sources = [inj_region] * n_mal + [host_region] * (N - n_mal)  # n_mal malicious copies, the rest clean
    copies = np.stack([noisy(s, p_err, rng) for s in sources])     # give every copy its own random errors
    return majority_consensus(copies)                              # vote
```
**In simple terms:** build N copies — `n_mal` of them start as the attacker's version, the rest start as the
true host — add sequencing noise to each one individually, then vote. This is one full "experiment": one
sequencing run, with a chosen number of copies and a chosen number of tainted ones.

```python
def diff_positions(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
```
**In simple terms:** the attacker's edit only changes some of the 1000 letters in a read (the "payload
footprint"). This just finds exactly which positions those are, by comparing a real clean read to its real
injected twin (from data we already generated in earlier steps). We only need to check the vote's answer at
*these* positions to know whether the attack survived.

## 3. The two different attacker "models" we tried, and why they gave opposite answers

This is the part that was confusing, so here it is explicitly.

### Model A — a *fixed number* of tainted copies (`ap_step10b_fixed_count_sweep.py`)

The attacker can only ever produce, say, exactly 10 tainted copies — no matter how many total copies (N) end
up sequenced. As N grows past 10, those 10 copies become a shrinking **share** of the vote (10 out of 20 is
50%, 10 out of 100 is only 10%). Eventually they're a small minority and lose.

**Result: malware survival goes DOWN as N goes up.** More sequencing depth *dilutes* a resource-limited
attacker. This is a real, useful defense against this kind of attacker.

### Model B — a fixed *probability* per copy (`ap_step10c_infection_probability.py`)

The attacker instead manages to taint **each** copy independently with some fixed probability — say 55%
(just over half), whatever N turns out to be. This is a believable model if, for example, the attacker
contaminates the *source material* the copies are made from, so every copy has the same 55% chance of coming
out tainted, independent of all the others.

Here, the realized number of tainted copies is random every time (see the math below) — and with only a few
copies, pure chance can easily land you at or below 50% tainted, even though the attacker's true rate is 55%.
With many copies, that randomness washes out and the realized fraction reliably ends up near 55% — a real
majority, almost every time.

**Result: malware survival goes UP as N goes up.** This is the opposite of Model A, and it's the one that
matches what your professor described.

Both are legitimate, realistic attacker capabilities — they just behave oppositely. That's the actual lesson:
**whether more sequencing depth helps or hurts depends on what kind of attacker you're facing**, not on some
universal rule about N.

## 4. The math behind Model B, in simple terms

### Coin flips

Each copy is like a coin flip: "tainted" with probability `p = 0.55`, "clean" with probability `0.45`. Flip
that coin N times (once per copy) and count how many times you get "tainted." That count is called a
**Binomial(N, p)** random variable — the standard way to describe "add up N independent yes/no coin flips."

**Tiny example, N = 3, p = 0.55.** There are 4 possible outcomes for the count of tainted copies (0, 1, 2, or
3), with these exact probabilities:

| Tainted copies | Probability |
|---|---|
| 0 | 0.45×0.45×0.45 = 9.1% |
| 1 | 3 × 0.55×0.45×0.45 = 33.4% |
| 2 | 3 × 0.55×0.55×0.45 = 40.8% |
| 3 | 0.55×0.55×0.55 = 16.6% |

(The "3 ×" counts the 3 different orders the one clean or one tainted copy could appear in among the 3.)

The attacker needs **more than half of 3**, i.e. 2 or 3 tainted, to win the vote: 40.8% + 16.6% = **57.5%
chance of winning** — even though their per-copy rate is 55%, not far off. This matches the simulation's
printed row for N=3 almost exactly.

### Why more copies make the attacker *more* reliable, not less

This is the **law of large numbers**: the more independent coin flips you add up, the closer the *fraction*
that come up "tainted" gets to the true probability (55%), and the less room there is for pure luck to push
the fraction below 50%.

**Rough version of the formula we use** (the "normal approximation" to a Binomial): with N copies and
per-copy probability p,
- average number tainted ≈ `N × p`
- "typical spread" (standard deviation) ≈ `sqrt(N × p × (1-p))`

As N grows, the average grows in direct proportion to N, but the spread only grows with the *square root* of
N — so the spread becomes tiny **relative to** the average. That's why the outcome becomes more and more
predictable (closer to a sure win for the attacker, since their average is already above 50%) as N increases.

**Worked example, N = 101, p = 0.55** (this is the row printed by the code):
- average tainted ≈ 101 × 0.55 = 55.55
- spread ≈ sqrt(101 × 0.55 × 0.45) ≈ 5.0
- the attacker needs more than 50.5 tainted to win
- that's (55.55 − 50.5) / 5.0 ≈ 1.0 "spread units" below the average
- from the standard normal table, that works out to about an **84% chance of winning** — matching the
  simulation's 84.7% almost exactly.

Compare that to N = 3 (57.5%) or N = 1 (where winning needs the single copy itself to be tainted, i.e.
exactly 55%). The win probability climbs steadily from ~55% toward ~100% as N grows — this is exactly the
rising red curve in the plot.

### The code for this part

```python
from scipy.stats import binom
exact = binom.sf(N // 2, N, P_TAINT)   # P(more than N//2 copies are tainted), the EXACT formula, no approximation
```
`binom.sf(k, N, p)` is a library function that computes "the probability of getting **more than** k successes
out of N independent trials with success probability p" — exactly, using the real Binomial formula (the
table above, generalized to any N), not the rough approximation. We used the approximation above only because
it's easier to compute by hand; the code itself uses the exact value.

```python
n_mal = int(rng.binomial(N, P_TAINT))   # SIMULATE: how many copies actually got tainted, this one time
cons = c.run_trial_count(hr, jr, N, n_mal, P_ERR, rng)
```
This is the simulation half: for every one of many repeated "trials," we roll a fresh random number of
tainted copies (this is what makes the simulated red dots in the plot land almost exactly on the thin exact
math line — two independent ways of answering the same question, agreeing with each other).

## 5. The plotting code

```python
def style(ax):
    ax.set_facecolor(BG)                 # light gray panel background, matching the rest of the project's figures
    ax.grid(color="white", lw=1)         # white gridlines on top of the gray
    ax.set_axisbelow(True)               # grid behind the data points, not on top of them
    for sp in ax.spines.values():
        sp.set_visible(False)            # remove the black box border
    ax.tick_params(length=0)             # remove the little tick marks
```
This is just cosmetics, reused from the project's other figures so everything looks consistent.

```python
ax.plot(Ns, [100 * v for v in exact_p], "-", color="#C44E52", alpha=0.5, label="Exact Binomial P(Win) ...")
ax.plot(Ns, [100 * v for v in survival], "o-", color="#C44E52", label="Simulated Malware Survival ...")
ax.plot(Ns, [100 * v for v in noise_rows], "s--", color="#4C72B0", label="Noise-Only Error ...")
```
Three lines, all sharing the same x-axis (`Ns`, the number of copies tried) and the same y-axis (a
percentage, 0–100): the thin faint line is the exact math answer, the solid line with circles is our
simulation's answer to the same question (they land on top of each other — good news, it means the
simulation is correct), and the dashed line with squares is the separate noise-correction experiment (no
attacker at all, just ordinary sequencing error), run across the same N values so it can be compared directly
on one plot.

## 6. The one-sentence takeaway

Ordinary sequencing noise always gets *better* with more copies (predictable, well understood). Whether
**malware** gets easier or harder to sneak through with more copies depends entirely on what kind of attacker
you assume: a resource-capped attacker (a fixed number of tainted copies) gets *diluted* by more copies, while
an attacker who can taint a steady fraction of whatever gets produced (a fixed probability per copy) gets
*more reliably successful* with more copies, for the same reason — the law of large numbers — that makes
noise correction reliable for the defender in the other scenario.
