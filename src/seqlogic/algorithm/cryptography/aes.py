"""Advanced Encryption Standard.

See https://csrc.nist.gov/pubs/fips/197/final for details.
"""

from bvwx import Array, Vec, bits, ite, rep, stack

NB = 4

Text = Vec[4 * 4 * 8]

# Nk = {4, 6, 8}
Key4 = Array[4, 4, 8]
Key6 = Array[6, 4, 8]
Key8 = Array[8, 4, 8]
Key = Key4 | Key6 | Key8

# Nr = {10, 12, 14}
RoundKeys = Array[11, 4, 4, 8] | Array[13, 4, 4, 8] | Array[15, 4, 4, 8]

Byte = Vec[8]
Word = Array[4, 8]
State = Array[4, 4, 8]
Polynomial = Array[4, 4]


# fmt: off

# Substitution Box
SBOX = bits([
    "8h63", "8h7C", "8h77", "8h7B", "8hF2", "8h6B", "8h6F", "8hC5",
    "8h30", "8h01", "8h67", "8h2B", "8hFE", "8hD7", "8hAB", "8h76",
    "8hCA", "8h82", "8hC9", "8h7D", "8hFA", "8h59", "8h47", "8hF0",
    "8hAD", "8hD4", "8hA2", "8hAF", "8h9C", "8hA4", "8h72", "8hC0",
    "8hB7", "8hFD", "8h93", "8h26", "8h36", "8h3F", "8hF7", "8hCC",
    "8h34", "8hA5", "8hE5", "8hF1", "8h71", "8hD8", "8h31", "8h15",
    "8h04", "8hC7", "8h23", "8hC3", "8h18", "8h96", "8h05", "8h9A",
    "8h07", "8h12", "8h80", "8hE2", "8hEB", "8h27", "8hB2", "8h75",

    "8h09", "8h83", "8h2C", "8h1A", "8h1B", "8h6E", "8h5A", "8hA0",
    "8h52", "8h3B", "8hD6", "8hB3", "8h29", "8hE3", "8h2F", "8h84",
    "8h53", "8hD1", "8h00", "8hED", "8h20", "8hFC", "8hB1", "8h5B",
    "8h6A", "8hCB", "8hBE", "8h39", "8h4A", "8h4C", "8h58", "8hCF",
    "8hD0", "8hEF", "8hAA", "8hFB", "8h43", "8h4D", "8h33", "8h85",
    "8h45", "8hF9", "8h02", "8h7F", "8h50", "8h3C", "8h9F", "8hA8",
    "8h51", "8hA3", "8h40", "8h8F", "8h92", "8h9D", "8h38", "8hF5",
    "8hBC", "8hB6", "8hDA", "8h21", "8h10", "8hFF", "8hF3", "8hD2",

    "8hCD", "8h0C", "8h13", "8hEC", "8h5F", "8h97", "8h44", "8h17",
    "8hC4", "8hA7", "8h7E", "8h3D", "8h64", "8h5D", "8h19", "8h73",
    "8h60", "8h81", "8h4F", "8hDC", "8h22", "8h2A", "8h90", "8h88",
    "8h46", "8hEE", "8hB8", "8h14", "8hDE", "8h5E", "8h0B", "8hDB",
    "8hE0", "8h32", "8h3A", "8h0A", "8h49", "8h06", "8h24", "8h5C",
    "8hC2", "8hD3", "8hAC", "8h62", "8h91", "8h95", "8hE4", "8h79",
    "8hE7", "8hC8", "8h37", "8h6D", "8h8D", "8hD5", "8h4E", "8hA9",
    "8h6C", "8h56", "8hF4", "8hEA", "8h65", "8h7A", "8hAE", "8h08",

    "8hBA", "8h78", "8h25", "8h2E", "8h1C", "8hA6", "8hB4", "8hC6",
    "8hE8", "8hDD", "8h74", "8h1F", "8h4B", "8hBD", "8h8B", "8h8A",
    "8h70", "8h3E", "8hB5", "8h66", "8h48", "8h03", "8hF6", "8h0E",
    "8h61", "8h35", "8h57", "8hB9", "8h86", "8hC1", "8h1D", "8h9E",
    "8hE1", "8hF8", "8h98", "8h11", "8h69", "8hD9", "8h8E", "8h94",
    "8h9B", "8h1E", "8h87", "8hE9", "8hCE", "8h55", "8h28", "8hDF",
    "8h8C", "8hA1", "8h89", "8h0D", "8hBF", "8hE6", "8h42", "8h68",
    "8h41", "8h99", "8h2D", "8h0F", "8hB0", "8h54", "8hBB", "8h16",
])

# Inverse Substitution Box
INV_SBOX = bits([
    "8h52", "8h09", "8h6A", "8hD5", "8h30", "8h36", "8hA5", "8h38",
    "8hBF", "8h40", "8hA3", "8h9E", "8h81", "8hF3", "8hD7", "8hFB",
    "8h7C", "8hE3", "8h39", "8h82", "8h9B", "8h2F", "8hFF", "8h87",
    "8h34", "8h8E", "8h43", "8h44", "8hC4", "8hDE", "8hE9", "8hCB",
    "8h54", "8h7B", "8h94", "8h32", "8hA6", "8hC2", "8h23", "8h3D",
    "8hEE", "8h4C", "8h95", "8h0B", "8h42", "8hFA", "8hC3", "8h4E",
    "8h08", "8h2E", "8hA1", "8h66", "8h28", "8hD9", "8h24", "8hB2",
    "8h76", "8h5B", "8hA2", "8h49", "8h6D", "8h8B", "8hD1", "8h25",

    "8h72", "8hF8", "8hF6", "8h64", "8h86", "8h68", "8h98", "8h16",
    "8hD4", "8hA4", "8h5C", "8hCC", "8h5D", "8h65", "8hB6", "8h92",
    "8h6C", "8h70", "8h48", "8h50", "8hFD", "8hED", "8hB9", "8hDA",
    "8h5E", "8h15", "8h46", "8h57", "8hA7", "8h8D", "8h9D", "8h84",
    "8h90", "8hD8", "8hAB", "8h00", "8h8C", "8hBC", "8hD3", "8h0A",
    "8hF7", "8hE4", "8h58", "8h05", "8hB8", "8hB3", "8h45", "8h06",
    "8hD0", "8h2C", "8h1E", "8h8F", "8hCA", "8h3F", "8h0F", "8h02",
    "8hC1", "8hAF", "8hBD", "8h03", "8h01", "8h13", "8h8A", "8h6B",

    "8h3A", "8h91", "8h11", "8h41", "8h4F", "8h67", "8hDC", "8hEA",
    "8h97", "8hF2", "8hCF", "8hCE", "8hF0", "8hB4", "8hE6", "8h73",
    "8h96", "8hAC", "8h74", "8h22", "8hE7", "8hAD", "8h35", "8h85",
    "8hE2", "8hF9", "8h37", "8hE8", "8h1C", "8h75", "8hDF", "8h6E",
    "8h47", "8hF1", "8h1A", "8h71", "8h1D", "8h29", "8hC5", "8h89",
    "8h6F", "8hB7", "8h62", "8h0E", "8hAA", "8h18", "8hBE", "8h1B",
    "8hFC", "8h56", "8h3E", "8h4B", "8hC6", "8hD2", "8h79", "8h20",
    "8h9A", "8hDB", "8hC0", "8hFE", "8h78", "8hCD", "8h5A", "8hF4",

    "8h1F", "8hDD", "8hA8", "8h33", "8h88", "8h07", "8hC7", "8h31",
    "8hB1", "8h12", "8h10", "8h59", "8h27", "8h80", "8hEC", "8h5F",
    "8h60", "8h51", "8h7F", "8hA9", "8h19", "8hB5", "8h4A", "8h0D",
    "8h2D", "8hE5", "8h7A", "8h9F", "8h93", "8hC9", "8h9C", "8hEF",
    "8hA0", "8hE0", "8h3B", "8h4D", "8hAE", "8h2A", "8hF5", "8hB0",
    "8hC8", "8hEB", "8hBB", "8h3C", "8h83", "8h53", "8h99", "8h61",
    "8h17", "8h2B", "8h04", "8h7E", "8hBA", "8h77", "8hD6", "8h26",
    "8hE1", "8h69", "8h14", "8h63", "8h55", "8h21", "8h0C", "8h7D",
])

# Round Constant
RCON = bits([
    "8h8D", "8h01", "8h02", "8h04", "8h08", "8h10", "8h20", "8h40",
    "8h80", "8h1B", "8h36", "8h6C", "8hD8", "8hAB", "8h4D", "8h9A",
    "8h2F", "8h5E", "8hBC", "8h63", "8hC6", "8h97", "8h35", "8h6A",
    "8hD4", "8hB3", "8h7D", "8hFA", "8hEF", "8hC5", "8h91", "8h39",
    "8h72", "8hE4", "8hD3", "8hBD", "8h61", "8hC2", "8h9F", "8h25",
    "8h4A", "8h94", "8h33", "8h66", "8hCC", "8h83", "8h1D", "8h3A",
    "8h74", "8hE8", "8hCB", "8h8D", "8h01", "8h02", "8h04", "8h08",
    "8h10", "8h20", "8h40", "8h80", "8h1B", "8h36", "8h6C", "8hD8",

    "8hAB", "8h4D", "8h9A", "8h2F", "8h5E", "8hBC", "8h63", "8hC6",
    "8h97", "8h35", "8h6A", "8hD4", "8hB3", "8h7D", "8hFA", "8hEF",
    "8hC5", "8h91", "8h39", "8h72", "8hE4", "8hD3", "8hBD", "8h61",
    "8hC2", "8h9F", "8h25", "8h4A", "8h94", "8h33", "8h66", "8hCC",
    "8h83", "8h1D", "8h3A", "8h74", "8hE8", "8hCB", "8h8D", "8h01",
    "8h02", "8h04", "8h08", "8h10", "8h20", "8h40", "8h80", "8h1B",
    "8h36", "8h6C", "8hD8", "8hAB", "8h4D", "8h9A", "8h2F", "8h5E",
    "8hBC", "8h63", "8hC6", "8h97", "8h35", "8h6A", "8hD4", "8hB3",

    "8h7D", "8hFA", "8hEF", "8hC5", "8h91", "8h39", "8h72", "8hE4",
    "8hD3", "8hBD", "8h61", "8hC2", "8h9F", "8h25", "8h4A", "8h94",
    "8h33", "8h66", "8hCC", "8h83", "8h1D", "8h3A", "8h74", "8hE8",
    "8hCB", "8h8D", "8h01", "8h02", "8h04", "8h08", "8h10", "8h20",
    "8h40", "8h80", "8h1B", "8h36", "8h6C", "8hD8", "8hAB", "8h4D",
    "8h9A", "8h2F", "8h5E", "8hBC", "8h63", "8hC6", "8h97", "8h35",
    "8h6A", "8hD4", "8hB3", "8h7D", "8hFA", "8hEF", "8hC5", "8h91",
    "8h39", "8h72", "8hE4", "8hD3", "8hBD", "8h61", "8hC2", "8h9F",

    "8h25", "8h4A", "8h94", "8h33", "8h66", "8hCC", "8h83", "8h1D",
    "8h3A", "8h74", "8hE8", "8hCB", "8h8D", "8h01", "8h02", "8h04",
    "8h08", "8h10", "8h20", "8h40", "8h80", "8h1B", "8h36", "8h6C",
    "8hD8", "8hAB", "8h4D", "8h9A", "8h2F", "8h5E", "8hBC", "8h63",
    "8hC6", "8h97", "8h35", "8h6A", "8hD4", "8hB3", "8h7D", "8hFA",
    "8hEF", "8hC5", "8h91", "8h39", "8h72", "8hE4", "8hD3", "8hBD",
    "8h61", "8hC2", "8h9F", "8h25", "8h4A", "8h94", "8h33", "8h66",
    "8hCC", "8h83", "8h1D", "8h3A", "8h74", "8hE8", "8hCB",
])

# a(x) = {03}x^3 + {01}x^2 + {01}x + {02}
PA = bits(["4h2", "4h1", "4h1", "4h3"])

# a^-1(x) = {0b}x^3 + {0d}x^2 + {09}x + {0e}
INV_PA = bits(["4hE", "4h9", "4hD", "4hB"])

# fmt: on


def sub_word(w: Word) -> Word:
    """AES SubWord() function.

    Function used in the Key Expansion routine that takes a four-byte input word
    and applies an S-box to each of the four bytes to produce an output word.
    """
    return stack(*[SBOX[b] for b in w])


def inv_sub_word(w: Word) -> Word:
    """AES InvSubWord() function.

    Transformation in the Inverse Cipher that is the inverse of SubBytes().
    """
    return stack(*[INV_SBOX[b] for b in w])


def rot_word(w: Word) -> Word:
    """AES RotWord() function.

    Function used in the Key Expansion routine that takes a four-byte word and
    performs a cyclic permutation.
    """
    return stack(w[1], w[2], w[3], w[0])


def xtime(b: Byte, n: int) -> Byte:
    """Repeated polynomial multiplication in GF(2^8)."""
    for _ in range(n):
        b = (b << 1) ^ ("8h1B" & rep(b[7], 8))
    return b


def rowxcol(a: Polynomial, b: Word) -> Byte:
    """Multiply one row and one column."""
    y = Byte.zeros()
    for i, a_i in enumerate(a):
        for j, a_ij in enumerate(a_i):
            y = ite(a_ij, y ^ xtime(b[3 - i], j), y)
    return y


def multiply(a: Polynomial, b: Word) -> Word:
    """Multiply two polynomials in GF(2^8).

    [a0 a1 a2 a3] x [b0 b1 b2 b3] = [d0 d1 d2 d3]

    [d0]   [a0 a3 a2 a1] [b0]
    [d1] = [a1 a0 a3 a2] [b1]
    [d2]   [a2 a1 a0 a3] [b2]
    [d3]   [a3 a2 a1 a0] [b3]
    """
    a0, a1, a2, a3 = a
    d = [
        rowxcol(stack(a1, a2, a3, a0), b),
        rowxcol(stack(a2, a3, a0, a1), b),
        rowxcol(stack(a3, a0, a1, a2), b),
        rowxcol(stack(a0, a1, a2, a3), b),
    ]
    return stack(*d)


def mix_columns(state: State) -> State:
    """AES MixColumns() function.

    Transformation in the Cipher that takes all of the columns of the State and
    mixes their data (independently of one another) to produce new columns.
    """
    return stack(*[multiply(PA, col) for col in state])


def inv_mix_columns(state: State) -> State:
    """AES InvMixColumns function.

    Transformation in the Inverse Cipher that is the inverse of MixColumns().
    """
    return stack(*[multiply(INV_PA, col) for col in state])


def sub_bytes(state: State) -> State:
    """AES SubBytes() function.

    Transformation in the Cipher that processes the State using a non­linear
    byte substitution table (S-box) that operates on each of the State bytes
    independently.
    """
    return stack(*[sub_word(col) for col in state])


def inv_sub_bytes(state: State) -> State:
    """AES InvSubBytes() function.

    Transformation in the Inverse Cipher that is the inverse of SubBytes().
    """
    return stack(*[inv_sub_word(col) for col in state])


def shift_rows(state: State) -> State:
    """AES ShiftRows() function.

    Transformation in the Cipher that processes the State by cyclically shifting
    the last three rows of the State by different offsets.
    """
    r0, r1, r2, r3 = (state[:, i] for i in range(4))
    cols = [
        stack(r0[0], r1[1], r2[2], r3[3]),
        stack(r0[1], r1[2], r2[3], r3[0]),
        stack(r0[2], r1[3], r2[0], r3[1]),
        stack(r0[3], r1[0], r2[1], r3[2]),
    ]
    return stack(*cols)


def inv_shift_rows(state: State) -> State:
    """AES InvShiftRows() function.

    Transformation in the Inverse Cipher that is the inverse of ShiftRows().
    """
    r0, r1, r2, r3 = (state[:, i] for i in range(4))
    cols = [
        stack(r0[0], r1[3], r2[2], r3[1]),
        stack(r0[1], r1[0], r2[3], r3[2]),
        stack(r0[2], r1[1], r2[0], r3[3]),
        stack(r0[3], r1[2], r2[1], r3[0]),
    ]
    return stack(*cols)


def key_expand(nk: int, k0: Word, k1: Word, i: int) -> Word:
    if i % nk == 0:
        rcon = bits([RCON[i // nk], "8h00", "8h00", "8h00"])
        return k0 ^ sub_word(rot_word(k1)) ^ rcon
    if nk > 6 and i % nk == 4:
        return k0 ^ sub_word(k1)
    return k0 ^ k1


def key_expansion(key: Key) -> RoundKeys:
    """Expand the key into the round key.

    See NIST FIPS 197 Section 5.2.
    """
    nk = key.shape[0]
    nr = nk + 6
    assert nk in (4, 6, 8)

    ws = list(key)
    for i in range(nk, NB * (nr + 1)):
        ws.append(key_expand(nk, ws[i - nk], ws[i - 1], i))

    # Convert {44, 52, 60} => {(10+1,4), (12+1,4), (14+1,4)}
    return stack(*ws).reshape(((nr + 1), 4, 4, 8))


def cipher(pt: Text, rkeys: RoundKeys) -> Text:
    """AES encryption cipher.

    See NIST FIPS 197 Section 5.1.
    """
    nr = rkeys.shape[0] - 1
    nk = nr - 6
    assert nk in (4, 6, 8)

    state = pt.reshape(State.shape)

    # first round
    state ^= rkeys[0]

    for i in range(1, nr):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state ^= rkeys[i]

    # final round
    state = sub_bytes(state)
    state = shift_rows(state)
    state ^= rkeys[nr]

    ct = state.reshape(Text.shape)
    return ct


def inv_cipher(ct: Text, rkeys: RoundKeys) -> Text:
    """AES decryption cipher.

    SEE NIST FIPS 197 Section 5.3.
    """
    nr = rkeys.shape[0] - 1
    nk = nr - 6
    assert nk in (4, 6, 8)

    state = ct.reshape(State.shape)

    # first round
    state ^= rkeys[nr]

    for i in range(nr - 1, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state ^= rkeys[i]
        state = inv_mix_columns(state)

    # final round
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state ^= rkeys[0]

    pt = state.reshape(Text.shape)
    return pt


def encrypt(pt: Text, key: Key) -> Text:
    """Encrypt a plain text block."""
    nk = key.shape[0]
    assert nk in (4, 6, 8)
    rkeys = key_expansion(key)
    return cipher(pt, rkeys)


def decrypt(ct: Text, key: Key) -> Text:
    """Decrypt a plain text block."""
    nk = key.shape[0]
    assert nk in (4, 6, 8)
    rkeys = key_expansion(key)
    return inv_cipher(ct, rkeys)
