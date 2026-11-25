"""Tests for ap_step3_encode.py: round trips, exact agreement with the authors' code, and the Fig. 3 check."""
import importlib.util
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ap_step3_encode as ap

# Load the authors' EncodeDecode class straight from their file (read-only; their file only imports `re`).
spec = importlib.util.spec_from_file_location("theirs", HERE.parent / "authors_repo/trojan_attack/encode_decode.py")
theirs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theirs)

random.seed(0)
msg = "john.smith:4821"

def tag_collision(msg, size, key):
    """True if the tag letters appear by chance inside some encoded fragment (decoder can't tell where it ends)."""
    return any(ap.START_TAG in ap.encode_fragment(t, i, key) for i, t in enumerate(ap.fragment(msg, size)))


print("1) Round trip: encode then decode gives back the message")
ok = collisions = 0
for size in range(1, 6):
    for key in (0, 10, 60):
        for r in range(0, 4):
            if tag_collision(msg, size, key):
                collisions += 1
                continue
            masked = ap.make_payload(msg, size, key, r)
            filled = masked.replace("*", "A")  # any host letter would do
            assert ap.read_payload(filled, key, r) == msg, (size, key, r)
            ok += 1
print(f"   {ok} combinations round-trip OK; {collisions} skipped because the tag appears inside the data")

# does the authors' decoder fail on the same collision case? (size 2, key 10)
ed = theirs.EncodeDecode(retain_pos=0)
p = ed.make_payload(msg, 2, encrypted=True, key=10)
print("   authors' decoder on the collision case (size 2, key 10):", repr(ed.get_payload(p, encrypted=True, key=10)))

print("2) Exact agreement with the authors' make_payload")
n = 0
for size in range(1, 6):
    for key in (0, 10, 20, 30, 40, 50, 60):
        for r in range(0, 6):
            mine = ap.make_payload(msg, size, key, r)
            cover = "".join(random.choice("ACGT") for _ in mine)
            mine_filled = "".join(c if m == "*" else m for c, m in zip(cover, mine))
            ed = theirs.EncodeDecode(retain_pos=r)
            enc = key != 0
            their = ed.make_payload(msg, size, encrypted=enc, cover_dna=cover, key=key)
            assert mine_filled == their, (size, key, r)
            n += 1
print(f"   OK on {n} parameter combinations")

print("3) Paper Fig. 3b check: fragment 'a1', serial number 0, key 60")
mine = ap.encode_fragment("a1", 0, 60)
print("   figure shows output bits: 1100 | 01101100 | 00001101  ->  TACGTAAATC")
bits = [format(0 ^ (60 & 0xF), "04b")] + [format(ord(c) ^ 60, "08b") for c in "a1"]
print("   we compute      bits:     ", " | ".join(bits), " ->", mine)
print("   'a' = 01100001, key = 00111100, XOR =", format(ord("a") ^ 60, "08b"))
