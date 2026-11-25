"""Step 3: turn a payload like 'john.smith:4821' into DNA letters.

Pipeline (paper Fig. 3):  fragment -> encrypt (XOR) + 2-bit encode -> add tags -> add retention.
'*' marks a retention slot: a position where the HOST read's own letter is kept.

WHO DID WHAT
  Authors (Islam et al. 2022): the whole idea and design: the five steps, the 2-bit A/C/G/T code, XOR encryption,
    the ACTG tags, retention, the 4-bit serial number, and the reference implementation in
    authors_repo/trojan_attack/encode_decode.py (class EncodeDecode).
  Us (ap_*): a from-scratch rewrite of that logic in ~80 lines of plain functions, so we understand it.
    Our additions: '*' masks (the authors use 'X' and need a cover DNA passed in), a simpler but equivalent XOR
    formula, and tests (ap_step3_test.py) that prove our output equals theirs on 210 settings, plus the
    tag-collision check and the Fig. 3b error check. Everything here follows the authors' design; none of it is a
    new method. `inject` is our own code for paper Eq. 1 and has NOT yet been compared against the authors' code.
"""
import re

BITS_TO_BASE = {"00": "A", "01": "C", "10": "G", "11": "T"}  # paper's encoding (authors' dnaDictionary)
BASE_TO_BITS = {b: bits for bits, b in BITS_TO_BASE.items()}
START_TAG = END_TAG = "ACTG"  # the authors' default tags in encode_decode.py


def bits_to_dna(bits):
    """'0110' -> 'CG': read the bits two at a time and swap each pair for a letter.
    Authors: same dictionary, spread across several loops in their code. Us: one small helper."""
    return "".join(BITS_TO_BASE[bits[i:i + 2]] for i in range(0, len(bits), 2))


def dna_to_bits(dna):
    """'CG' -> '0110': the reverse of bits_to_dna. Used when decoding."""
    return "".join(BASE_TO_BITS[b] for b in dna)


def fragment(msg, size):
    """'a17wit' -> ['a1', '7w', 'it'] for size 2: cut the message into pieces of `size` characters.
    Authors: make_fragments uses a regex for this. Us: plain slicing, same result."""
    return [msg[i:i + size] for i in range(0, len(msg), size)]


def encode_fragment(text, order, key):
    """Encrypt one fragment and turn it into DNA letters.
    4-bit serial number + 8 bits per character, each XORed with the key, then 2 bits -> 1 letter.
    The serial number only gets the low 4 bits of the key (it is 4 bits wide). key=0 means no encryption.
    Authors: fragment_encrypted_encode builds an 8-bit serial byte, XORs it, then chops off 4 bits.
    Us: apply the XOR to the 4 bits directly, which gives the identical result (verified by the tests)."""
    bits = format(order ^ (key & 0xF), "04b")
    bits += "".join(format(ord(c) ^ key, "08b") for c in text)
    return bits_to_dna(bits)


def decode_fragment(dna, key):
    """Undo encode_fragment: letters -> bits -> XOR with the key again -> (serial number, text).
    XOR twice with the same key returns the original, so decryption is the same operation as encryption.
    Authors: fragment_decrypt_decode. Us: rewritten from scratch, same behaviour."""
    bits = dna_to_bits(dna)
    order = int(bits[:4], 2) ^ (key & 0xF)
    text = "".join(chr(int(bits[i:i + 8], 2) ^ key) for i in range(4, len(bits), 8))
    return order, text


def make_fragments(msg, fragment_size, key=60, retention=0):
    """The attacker-side pipeline, one string per fragment: cut, encrypt+encode, add start/end tags, add retention slots.
    Retention slots are '*' (host letter kept there). Step 4b needs the list so each fragment can be placed separately.
    Authors: make_fragments in encode_decode.py (they use 'X' instead of '*'). Us: same logic, '*' mask."""
    pieces = []
    for order, text in enumerate(fragment(msg, fragment_size)):
        piece = START_TAG + encode_fragment(text, order, key) + END_TAG
        # retention: put `retention` '*' slots in front of every letter
        pieces.append("".join("*" * retention + c for c in piece))
    return pieces


def make_payload(msg, fragment_size, key=60, retention=0):
    """All fragments joined into one block (what step 3 and its tests use).
    Authors: make_payload, which also needs the cover DNA passed in and fills slots with it immediately.
    Us: return the '*' mask and let `inject` do the filling, so the same payload can be placed anywhere."""
    return "".join(make_fragments(msg, fragment_size, key, retention))


def read_payload(dna, key=60, retention=0):
    """The Trojan side: scan a DNA string for tagged fragments and rebuild the message.
    Needs the same retention and key as the attacker used.
    Authors: get_fragments + get_payload (regex search for the tags, drop retention letters, decrypt, sort by serial).
    Us: the same approach, shorter. Known weakness inherited from the design: if the tag letters happen to appear
    inside an encoded fragment, the fragment is cut short (see tag_collision in ap_step3_test.py)."""
    gap = "[ACGTN]" * retention
    tag = lambda t: "".join(gap + c for c in t)
    pattern = re.compile(tag(START_TAG) + "(.*?)" + tag(END_TAG))
    found = {}
    pos = 0
    while (m := pattern.search(dna, pos)):
        body = m.group(1)
        real = body[retention::retention + 1]  # drop the retention letters, keep the real ones
        order, text = decode_fragment(real, key)
        found[order] = text
        pos = m.end()
    return "".join(found[i] for i in sorted(found))


def read_payload_robust(dna, key=60, retention=0, valid=r"[a-z0-9]+"):
    """A smarter Trojan: try EVERY start-tag occurrence (not just the first), pair it with the next end tag, decode, and keep only
    fragments whose serial number is 0-15 and whose text matches `valid` (the attacker knows the payload alphabet).
    Fixes the naive decoder's failure when a start tag also occurs naturally in the host. Us only: the authors' decoder scans naively.
    Assumes the payload alphabet is lowercase letters and digits (true for the authors' payloads); change `valid` otherwise."""
    gap = "[ACGTN]" * retention
    tag = lambda t: "".join(gap + c for c in t)
    start_len = len(START_TAG) * (retention + 1)  # letters the start tag really covers in the read (not the regex text length)
    end_pat = re.compile(tag(END_TAG))
    found = {}
    for m in re.finditer("(?=" + tag(START_TAG) + ")", dna):
        body_start = m.start() + start_len
        e = end_pat.search(dna, body_start)
        if not e:
            continue
        real = dna[body_start:e.start()][retention::retention + 1]
        if len(real) < 6 or len(real) % 2:  # need serial (2 letters) + whole characters (4 letters each)
            continue
        order, text = decode_fragment(real, key)
        if 0 <= order < 16 and re.fullmatch(valid, text) and order not in found:
            found[order] = text
    return "".join(found[i] for i in sorted(found))


def inject(host, payload_masked, i):
    """Paper Eq. 1: overwrite the host read starting at position i with the payload; '*' keeps the host letter.
    Authors: do this inside their insertion code (trojan_experiment_utility.py / nw_insertion.py), together with the
    Needleman-Wunsch search for the best i. Us: only the plain substitution for now; the NW search is Step 4b."""
    out = list(host)
    for j, c in enumerate(payload_masked):
        if c != "*":
            out[i + j] = c
    return "".join(out)
