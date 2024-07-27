"""Test AES Algorithm."""

from seqlogic import Vector, uint2vec
from seqlogic.algorithms.cryptography.aes import (
    Key4,
    Key6,
    Key8,
    Text,
    decrypt,
    encrypt,
    key_expansion,
)


def _s2v(s: str) -> Vector:
    a = bytearray.fromhex(s)
    return uint2vec(int.from_bytes(a, byteorder="little"), 4 * len(s))


# fmt: off
A1_EXP = [
    0x16157E2B, 0xA6D2AE28, 0x8815F7AB, 0x3C4FCF09,
    0x17FEFAA0, 0xB12C5488, 0x3939A323, 0x05766C2A,
    0xF295C2F2, 0x43B9967A, 0x7A803559, 0x7FF65973,
    0x7D47803D, 0x3EFE1647, 0x447E231E, 0x3B887A6D,
    0x41A544EF, 0x7F5B52A8, 0x3B2571B6, 0x00AD0BDB,
    0xF8C6D1D4, 0x879D837C, 0xBCB8F2CA, 0xBC15F911,
    0x7AA3886D, 0xFD3E0B11, 0x4186F9DB, 0xFD9300CA,
    0x0EF7544E, 0xF3C95F5F, 0xB24FA684, 0x4FDCA64E,
    0x2173D2EA, 0xD2BA8DB5, 0x60F52B31, 0x2F298D7F,
    0xF36677AC, 0x21DCFA19, 0x4129D128, 0x6E005C57,
    0xA8F914D0, 0x8925EEC9, 0xC80C3FE1, 0xA60C63B6,
]
# fmt: on


def test_a1():
    """Test using values from Appendix A.1."""
    v = _s2v("2b7e151628aed2a6abf7158809cf4f3c")
    key = v.reshape(Key4.shape)
    rkeys = key_expansion(key)
    for i, rkey in enumerate(rkeys):
        for j, w in enumerate(rkey):
            k = rkey.shape[0] * i + j
            assert str(w.flatten()) == f"32b{A1_EXP[k]:039_b}"


# fmt: off
A2_EXP = [
    0xF7B0738E, 0x52640EDA, 0x2BF310C8, 0xE5799080,
    0xD2EAF862, 0x7B6B2C52, 0xF7910CFE, 0xA5F50224,
    0x8E0612EC, 0x6B7F826C, 0xB9957A0E, 0xC2FE565C,
    0xBDB4B74D, 0x1841B569, 0x9647A785, 0xFD3825E9,
    0x44AD5FE7, 0x865309BB, 0x57F05A48, 0x4FB1EF21,
    0xD9F648A4, 0x24CE6D4D, 0x606332AA, 0xE6303B11,
    0xD57E5EA2, 0x9ACFB183, 0x4339F927, 0x67F7946A,
    0x0794A6C0, 0xE1A49DD1, 0xEB8617EC, 0x7149A66F,
    0x32705F48, 0x5587CB22, 0x52136DE2, 0xB3B7F033,
    0x28EBBE40, 0x59A2182F, 0x6BD24767, 0x3E558C45,
    0x6C46E1A7, 0xDFF11194, 0x0A751F82, 0x53D707AD,
    0x380540CA, 0x0650CC8F, 0x6A162D28, 0xB5E73CBC,
    0x6FA08BE9, 0x3C778C44, 0x0472CC8E, 0x02220001,
]
# fmt: on


def test_a2():
    """Test using values from Appendix A.2."""
    v = _s2v("8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b")
    key = v.reshape(Key6.shape)
    rkeys = key_expansion(key)
    for i, rkey in enumerate(rkeys):
        for j, w in enumerate(rkey):
            k = rkey.shape[0] * i + j
            assert str(w.flatten()) == f"32b{A2_EXP[k]:039_b}"


# fmt: off
A3_EXP = [
    0x10EB3D60, 0xBE71CA15, 0xF0AE732B, 0x81777D85,
    0x072C351F, 0xD708613B, 0xA310982D, 0xF4DF1409,
    0x1154A39B, 0xAF25698E, 0x5F8B1AA5, 0xDEFC6720,
    0x1A9CB0A8, 0xCD94D193, 0x6E8449BE, 0x9A5B5DB7,
    0xB8EC9AD5, 0x17C9F35B, 0x4842E9FE, 0x96BE8EDE,
    0x8A32A9B5, 0x47A67826, 0x29223198, 0xB3796C2F,
    0xAD812C81, 0xBA48DFDA, 0xF20A3624, 0x64B4B8FA,
    0xC9BFC598, 0x8E19BDBE, 0xA73B8C26, 0x1442E009,
    0xAC7B0068, 0x1633DFB2, 0xE439E996, 0x808D516C,
    0x04E214C8, 0x8AFBA976, 0x2DC02550, 0x3982C559,
    0x676913DE, 0x715ACC6C, 0x956325FA, 0x15EE7496,
    0x5DCA8658, 0xD7312F2E, 0xFAF10A7E, 0xC373CF27,
    0xAB479C74, 0xDA1D5018, 0x4F7E75E2, 0x5A900174,
    0xE3AAFACA, 0x349BD5E4, 0xCE6ADF9A, 0x0D1910BD,
    0xD19048FE, 0x0B8D18E6, 0x44F36D04, 0x1E636C70,
]
# fmt: on


def test_a3():
    """Test using values from Appendix A.3."""
    v = _s2v("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    key = v.reshape(Key8.shape)
    rkeys = key_expansion(key)
    for i, rkey in enumerate(rkeys):
        for j, w in enumerate(rkey):
            k = rkey.shape[0] * i + j
            assert str(w.flatten()) == f"32b{A3_EXP[k]:039_b}"


PT = _s2v("00112233445566778899aabbccddeeff").reshape(Text.shape)


def test_c1():
    """Test using values from Appendix C.1."""
    v = _s2v("000102030405060708090a0b0c0d0e0f")
    key = v.reshape(Key4.shape)
    ct_got = encrypt(PT, key)

    ct = _s2v("69c4e0d86a7b0430d8cdb78070b4c55a").reshape(Text.shape)
    assert ct_got == ct

    pt_got = decrypt(ct, key)
    assert pt_got == PT


def test_c2():
    """Test using values from Appendix C.2."""
    v = _s2v("000102030405060708090a0b0c0d0e0f1011121314151617")
    key = v.reshape(Key6.shape)
    ct_got = encrypt(PT, key)

    ct = _s2v("dda97ca4864cdfe06eaf70a0ec0d7191").reshape(Text.shape)
    assert ct_got == ct

    pt_got = decrypt(ct, key)
    assert pt_got == PT


def test_c3():
    """Test using values from Appendix C.3."""
    v = _s2v("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
    key = v.reshape(Key8.shape)
    ct_got = encrypt(PT, key)

    ct = _s2v("8ea2b7ca516745bfeafc49904b496089").reshape(Text.shape)
    assert ct_got == ct

    pt_got = decrypt(ct, key)
    assert pt_got == PT
