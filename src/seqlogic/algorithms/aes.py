"""
Advanced Encryption Standard

See https://csrc.nist.gov/pubs/fips/197/final for details.
"""

from collections import deque

from ..logic import logic
from ..logicvec import cat, logicvec, rep, uint2vec

NB = 4

_BYTE_BITS = 8
_WORD_BYTES = 4
_WORD_BITS = _WORD_BYTES * _BYTE_BITS
_BYTE_SHAPE = (_BYTE_BITS,)
_TEXT_SHAPE = (NB * _WORD_BITS,)


# fmt: off
_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

_INV_SBOX = [
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
]

_RCON = [
    0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
    0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A,
    0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8,
    0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF,
    0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC,
    0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B,
    0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3,
    0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94,
    0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
    0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35,
    0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F,
    0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04,
    0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63,
    0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD,
    0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB,
]

_MTXA = [
    0x2311,
    0x1231,
    0x1123,
    0x3112,
]

_INV_MTXA = [
    0xEBD9,
    0x9EBD,
    0xD9EB,
    0xBD9E,
]
# fmt: on


# Convert raw data to logicvec
SBOX = cat([uint2vec(x, _BYTE_BITS) for x in _SBOX])
INV_SBOX = cat([uint2vec(x, _BYTE_BITS) for x in _INV_SBOX])
RCON = cat([uint2vec(x, _BYTE_BITS) for x in _RCON])
MTXA = cat([uint2vec(x, 16) for x in _MTXA])
INV_MTXA = cat([uint2vec(x, 16) for x in _INV_MTXA])


def sub_word(w: logicvec) -> logicvec:
    """
    Function used in the Key Expansion routine that takes a four-byte input word
    and applies an S-box to each of the four bytes to produce an output word.
    """
    w = w.reshape((_WORD_BYTES, _BYTE_BITS))
    return cat([SBOX[w[b]] for b in range(_WORD_BYTES)], flatten=True)


def inv_sub_word(w: logicvec) -> logicvec:
    """
    Transformation in the Inverse Cipher that is the inverse of SubBytes().
    """
    w = w.reshape((_WORD_BYTES, _BYTE_BITS))
    return cat([INV_SBOX[w[b]] for b in range(_WORD_BYTES)], flatten=True)


def rot_word(w: logicvec) -> logicvec:
    """
    Function used in the Key Expansion routine that takes a four-byte word and
    performs a cyclic permutation.
    """
    w = w.reshape((_WORD_BYTES, _BYTE_BITS))
    bytes_ = deque(w[b] for b in range(_WORD_BYTES))
    bytes_.rotate(-1)
    return cat(bytes_, flatten=True)


def xtime(b: logicvec, n: int) -> logicvec:
    """Repeated polynomial multiplication in GF(2^8)."""
    b = b.reshape(_BYTE_SHAPE)
    for _ in range(n):
        b = (b << 1) ^ (uint2vec(0x1B, _BYTE_BITS) & rep(b[7], _BYTE_BITS))
    return b


def _rowxcol(row: logicvec, col: logicvec) -> logicvec:
    """Multiply one row and one column."""
    row = row.reshape((4, 4))
    col = col.reshape((_WORD_BYTES, _BYTE_BITS))

    y = uint2vec(0, _BYTE_BITS)
    for i in range(4):
        for j in range(4):
            if row[i, j] is logic.T:
                y ^= xtime(col[3 - i], j)

    return y


def _multiply(a: logicvec, col: logicvec) -> logicvec:
    """Multiply a matrix by one column."""
    a = a.reshape((4, 4, 4))
    col = col.reshape((_WORD_BYTES, _BYTE_BITS))
    return cat([_rowxcol(a[c], col) for c in range(NB)])


def mix_columns(state: logicvec) -> logicvec:
    """
    Transformation in the Cipher that takes all of the columns of the State and
    mixes their data (independently of one another) to produce new columns.
    """
    state = state.reshape((NB, _WORD_BYTES, _BYTE_BITS))
    return cat([_multiply(MTXA, state[c]) for c in range(NB)])


def inv_mix_columns(state: logicvec) -> logicvec:
    """
    Transformation in the Inverse Cipher that is the inverse of MixColumns().
    """
    state = state.reshape((NB, _WORD_BYTES, _BYTE_BITS))
    return cat([_multiply(INV_MTXA, state[c]) for c in range(NB)])


def add_round_key(state: logicvec, rkey: logicvec) -> logicvec:
    """
    Transformation in the Cipher and Inverse Cipher in which a Round Key is
    added to the State using an XOR operation. The length of a Round Key equals
    the size of the State (i.e., for Nb = 4, the Round Key length equals 128
    bits/16 bytes).
    """
    state = state.reshape((NB, _WORD_BITS))
    rkey = rkey.reshape((NB, _WORD_BITS))
    words = [state[c] ^ rkey[c] for c in range(NB)]
    return cat(words)


def sub_bytes(state: logicvec) -> logicvec:
    """
    Transformation in the Cipher that processes the State using a non­linear
    byte substitution table (S-box) that operates on each of the State bytes
    independently.
    """
    state = state.reshape((NB, _WORD_BITS))
    words = [sub_word(state[c]) for c in range(NB)]
    return cat(words)


def inv_sub_bytes(state: logicvec) -> logicvec:
    """
    Transformation in the Inverse Cipher that is the inverse of SubBytes().
    """
    state = state.reshape((NB, _WORD_BITS))
    words = [inv_sub_word(state[c]) for c in range(NB)]
    return cat(words)


def shift_rows(state: logicvec) -> logicvec:
    """
    Transformation in the Cipher that processes the State by cyclically shifting
    the last three rows of the State by different offsets.
    """
    state = state.reshape((NB, _WORD_BYTES, _BYTE_BITS))

    bytes_ = []
    cs = deque(range(NB))
    for _ in range(NB):
        for r in range(4):
            bytes_.append(state[cs[r], r])
        cs.rotate(-1)

    return cat(bytes_)


def inv_shift_rows(state: logicvec) -> logicvec:
    """
    Transformation in the Inverse Cipher that is the inverse of ShiftRows().
    """
    state = state.reshape((NB, _WORD_BYTES, _BYTE_BITS))

    bytes_ = []
    cs = deque(reversed(range(NB)))
    for _ in range(NB):
        cs.rotate(1)
        for r in range(4):
            bytes_.append(state[cs[r], r])

    return cat(bytes_)


def key_expansion(nk: int, key: logicvec) -> logicvec:
    """Expand the key into the round key."""
    assert nk in (4, 6, 8)

    nr = nk + 6
    key = key.reshape((nk, _WORD_BITS))

    w = [key[k] for k in range(nk)]
    for k in range(nk, NB * (nr + 1)):
        temp = w[k - 1]
        if k % nk == 0:
            temp = sub_word(rot_word(temp)) ^ RCON[k // nk].zext(_BYTE_BITS * 3)
        elif nk > 6 and k % nk == 4:
            temp = sub_word(temp)
        w.append(w[k - nk] ^ temp)

    return cat(w)


def _rksl(k: int) -> slice:
    return slice(NB * k, NB * (k + 1))


def cipher(nk: int, pt: logicvec, rkey: logicvec):
    """
    AES encryption cipher.

    See NIST FIPS 197 Section 5.3
    """
    assert nk in (4, 6, 8)

    nr = nk + 6
    rkey = rkey.reshape((NB * (nr + 1), _WORD_BITS))

    # first round
    state = add_round_key(pt, rkey[_rksl(0)])

    for k in range(1, nr):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, rkey[_rksl(k)])

    # final round
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, rkey[_rksl(nr)])

    return state.reshape(_TEXT_SHAPE)


def inv_cipher(nk: int, ct: logicvec, rkey: logicvec):
    """
    AES decryption cipher.
    """
    assert nk in (4, 6, 8)

    nr = nk + 6
    rkey = rkey.reshape((NB * (nr + 1), _WORD_BITS))

    # first round
    state = add_round_key(ct, rkey[_rksl(nr)])

    for k in range(nr - 1, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, rkey[_rksl(k)])
        state = inv_mix_columns(state)

    # final round
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, rkey[_rksl(0)])

    return state.reshape(_TEXT_SHAPE)


def encrypt(nk: int, pt: logicvec, key: logicvec) -> logicvec:
    """Encrypt a plain text block."""
    assert nk in (4, 6, 8)
    rkey = key_expansion(nk, key)
    return cipher(nk, pt, rkey)


def decrypt(nk: int, ct: logicvec, key: logicvec) -> logicvec:
    """Decrypt a plain text block."""
    assert nk in (4, 6, 8)
    rkey = key_expansion(nk, key)
    return inv_cipher(nk, ct, rkey)