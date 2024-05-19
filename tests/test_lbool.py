"""Test seqlogic.lbool module."""

# pylint: disable = pointless-statement
# pylint: disable = protected-access

import pytest

from seqlogic import lbool
from seqlogic.lbool import (
    Vec,
    and_,
    cat,
    dcs,
    implies,
    int2vec,
    nand,
    nor,
    not_,
    or_,
    rep,
    uint2vec,
    vec,
    xes,
    xnor,
    xor,
)

E = Vec[0](0)
X = Vec[1](0b00)
F = Vec[1](0b01)
T = Vec[1](0b10)
W = Vec[1](0b11)


def test_vec():
    # None/Empty
    assert vec() == vec(None) == vec([]) == E
    # Single bool input
    assert vec(False) == vec(0) == F
    assert vec(True) == vec(1) == T
    # Sequence of bools
    assert vec([False, True, 0, 1]) == Vec[4](0b10_01_10_01)
    # String
    assert vec("4b-10X") == Vec[4](0b11_10_01_00)

    # Invalid input type
    with pytest.raises(TypeError):
        vec({"key": "val"})


LNOT = {
    "X": "X",
    "0": "1",
    "1": "0",
    "-": "-",
}


def test_lnot():
    for x, y in LNOT.items():
        x = lbool._from_char[x]
        y = lbool._from_char[y]
        assert not_(x) == y


NOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "1",
    "01": "0",
    "0-": "-",
    "1X": "X",
    "10": "0",
    "11": "0",
    "1-": "0",
    "-X": "X",
    "-0": "-",
    "-1": "0",
    "--": "-",
}


def test_lnor():
    for xs, y in NOR.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert nor(x0, x1) == y


OR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "0",
    "01": "1",
    "0-": "-",
    "1X": "X",
    "10": "1",
    "11": "1",
    "1-": "1",
    "-X": "X",
    "-0": "-",
    "-1": "1",
    "--": "-",
}


def test_lor():
    for xs, y in OR.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert or_(x0, x1) == y


NAND = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "1",
    "01": "1",
    "0-": "1",
    "1X": "X",
    "10": "1",
    "11": "0",
    "1-": "-",
    "-X": "X",
    "-0": "1",
    "-1": "-",
    "--": "-",
}


def test_lnand():
    for xs, y in NAND.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert nand(x0, x1) == y


AND = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "0",
    "01": "0",
    "0-": "0",
    "1X": "X",
    "10": "0",
    "11": "1",
    "1-": "-",
    "-X": "X",
    "-0": "0",
    "-1": "-",
    "--": "-",
}


def test_land():
    for xs, y in AND.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert and_(x0, x1) == y


XNOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "1",
    "01": "0",
    "0-": "-",
    "1X": "X",
    "10": "0",
    "11": "1",
    "1-": "-",
    "-X": "X",
    "-0": "-",
    "-1": "-",
    "--": "-",
}


def test_lxnor():
    for xs, y in XNOR.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert xnor(x0, x1) == y


XOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "0",
    "01": "1",
    "0-": "-",
    "1X": "X",
    "10": "1",
    "11": "0",
    "1-": "-",
    "-X": "X",
    "-0": "-",
    "-1": "-",
    "--": "-",
}


def test_lxor():
    for xs, y in XOR.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert xor(x0, x1) == y


IMPLIES = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "X-": "X",
    "0X": "X",
    "00": "1",
    "01": "1",
    "0-": "1",
    "1X": "X",
    "10": "0",
    "11": "1",
    "1-": "-",
    "-X": "X",
    "-0": "-",
    "-1": "1",
    "--": "-",
}


def test_limplies():
    for xs, y in IMPLIES.items():
        x0 = lbool._from_char[xs[0]]
        x1 = lbool._from_char[xs[1]]
        y = lbool._from_char[y]
        assert implies(x0, x1) == y


def test_vec_cls_getitem():
    vec_4 = Vec[4]
    v = vec_4(42)  # pyright: ignore[reportCallIssue]
    assert len(v) == 4 and v.data == 42
    # Always return the same class instance
    assert Vec[4] is vec_4


def test_vec_repr():
    assert repr(Vec[0](0)) == "Vec[0](0b0)"
    assert repr(Vec[4](0b1110_0100)) == "Vec[4](0b11100100)"


def test_vec_bool():
    assert bool(Vec[0](0)) is False
    assert bool(Vec[1](0b01)) is False
    assert bool(Vec[1](0b10)) is True
    assert bool(Vec[4](0b01_01_01_01)) is False
    assert bool(Vec[4](0b10_01_10_01)) is True
    with pytest.raises(ValueError):
        bool(Vec[4](0b11_10_01_00))


def test_vec_int():
    assert int(Vec[0](0)) == 0
    assert int(Vec[1](0b01)) == 0
    assert int(Vec[1](0b10)) == -1
    assert int(Vec[4](0b01_01_01_01)) == 0
    assert int(Vec[4](0b10_01_10_01)) == -6
    assert int(Vec[4](0b01_10_01_10)) == 5
    with pytest.raises(ValueError):
        int(Vec[4](0b11_10_01_00))


def test_vec_eq():
    assert Vec[0](0) == Vec[0](0)
    assert Vec[4](0b10_01_10_01) == Vec[4](0b10_01_10_01)
    assert Vec[4](0b10_01_10_01) != Vec[4](0b01_10_01_10)
    assert Vec[4](0b10_01_10_01) == "4b1010"
    assert Vec[4](0b10_01_10_01) != "4b0101"
    assert Vec[4](0b10_01_10_01) == "4b----"


def test_vec_hash():
    s = set()
    s.add(lbool.uint2vec(0))
    s.add(lbool.uint2vec(1))
    s.add(lbool.uint2vec(2))
    s.add(lbool.uint2vec(3))
    s.add(lbool.uint2vec(1))
    s.add(lbool.uint2vec(2))
    assert len(s) == 4


def test_vec_lnot():
    x = Vec[4](0b11_10_01_00)
    assert x.not_() == Vec[4](0b11_01_10_00)
    assert ~x == Vec[4](0b11_01_10_00)


def test_vec_lnor():
    x0 = Vec[16](0b11111111_10101010_01010101_00000000)
    x1 = Vec[16](0b11100100_11100100_11100100_11100100)
    assert x0.nor(x1) == Vec[16](0b11011100_01010100_11011000_00000000)
    assert ~(x0 | x1) == Vec[16](0b11011100_01010100_11011000_00000000)


def test_vec_lor():
    x0 = Vec[16](0b11111111_10101010_01010101_00000000)
    x1 = Vec[16](0b11100100_11100100_11100100_11100100)
    assert x0.or_(x1) == Vec[16](0b11101100_10101000_11100100_00000000)
    assert x0 | x1 == Vec[16](0b11101100_10101000_11100100_00000000)


def test_vec_lnand():
    x0 = Vec[16](0b11111111_10101010_01010101_00000000)
    x1 = Vec[16](0b11100100_11100100_11100100_11100100)
    assert x0.nand(x1) == Vec[16](0b11111000_11011000_10101000_00000000)
    assert ~(x0 & x1) == Vec[16](0b11111000_11011000_10101000_00000000)


def test_vec_land():
    x0 = Vec[16](0b11111111_10101010_01010101_00000000)
    x1 = Vec[16](0b11100100_11100100_11100100_11100100)
    assert x0.and_(x1) == Vec[16](0b11110100_11100100_01010100_00000000)
    assert (x0 & x1) == Vec[16](0b11110100_11100100_01010100_00000000)


def test_vec_lxnor():
    x0 = Vec[16](0b11111111_10101010_01010101_00000000)
    x1 = Vec[16](0b11100100_11100100_11100100_11100100)
    assert x0.xnor(x1) == Vec[16](0b11111100_11100100_11011000_00000000)
    assert ~(x0 ^ x1) == Vec[16](0b11111100_11100100_11011000_00000000)


def test_vec_lxor():
    x0 = Vec[16](0b11111111_10101010_01010101_00000000)
    x1 = Vec[16](0b11100100_11100100_11100100_11100100)
    assert x0.xor(x1) == Vec[16](0b11111100_11011000_11100100_00000000)
    assert (x0 ^ x1) == Vec[16](0b11111100_11011000_11100100_00000000)

    # Vector length mismatch
    with pytest.raises(TypeError):
        x0.xor(Vec[8](0b11111111_00000000))


UOR = {
    0b00_00: 0b00,
    0b01_00: 0b00,
    0b10_00: 0b00,
    0b11_00: 0b00,
    0b00_01: 0b00,
    0b01_01: 0b01,
    0b10_01: 0b10,
    0b11_01: 0b11,
    0b00_10: 0b00,
    0b01_10: 0b10,
    0b10_10: 0b10,
    0b11_10: 0b10,
    0b00_11: 0b00,
    0b01_11: 0b11,
    0b10_11: 0b10,
    0b11_11: 0b11,
}


def test_vec_ulor():
    for k, v in UOR.items():
        assert Vec[2](k).uor() == Vec[1](v)


UAND = {
    0b00_00: 0b00,
    0b01_00: 0b00,
    0b10_00: 0b00,
    0b11_00: 0b00,
    0b00_01: 0b00,
    0b01_01: 0b01,
    0b10_01: 0b01,
    0b11_01: 0b01,
    0b00_10: 0b00,
    0b01_10: 0b01,
    0b10_10: 0b10,
    0b11_10: 0b11,
    0b00_11: 0b00,
    0b01_11: 0b01,
    0b10_11: 0b11,
    0b11_11: 0b11,
}


def test_vec_uand():
    for k, v in UAND.items():
        assert Vec[2](k).uand() == Vec[1](v)


UXNOR = {
    0b00_00: 0b00,
    0b01_00: 0b00,
    0b10_00: 0b00,
    0b11_00: 0b00,
    0b00_01: 0b00,
    0b01_01: 0b10,
    0b10_01: 0b01,
    0b11_01: 0b11,
    0b00_10: 0b00,
    0b01_10: 0b01,
    0b10_10: 0b10,
    0b11_10: 0b11,
    0b00_11: 0b00,
    0b01_11: 0b11,
    0b10_11: 0b11,
    0b11_11: 0b11,
}


def test_vec_uxnor():
    for k, v in UXNOR.items():
        assert Vec[2](k).uxnor() == Vec[1](v)


UXOR = {
    0b00_00: 0b00,
    0b01_00: 0b00,
    0b10_00: 0b00,
    0b11_00: 0b00,
    0b00_01: 0b00,
    0b01_01: 0b01,
    0b10_01: 0b10,
    0b11_01: 0b11,
    0b00_10: 0b00,
    0b01_10: 0b10,
    0b10_10: 0b01,
    0b11_10: 0b11,
    0b00_11: 0b00,
    0b01_11: 0b11,
    0b10_11: 0b11,
    0b11_11: 0b11,
}


def test_vec_ltu():
    zero = uint2vec(0, 8)
    one = uint2vec(1, 8)
    two = uint2vec(2, 8)
    assert not zero.ltu(zero)
    assert zero.ltu(one)
    assert zero.ltu(two)
    assert not one.ltu(zero)
    assert not one.ltu(one)
    assert one.ltu(two)
    assert not two.ltu(zero)
    assert not two.ltu(one)
    assert not two.ltu(two)


def test_vec_lt():
    n_one = int2vec(-1, 8)
    zero = int2vec(0, 8)
    one = int2vec(1, 8)
    assert not n_one.lt(n_one)
    assert n_one.lt(zero)
    assert n_one.lt(one)
    assert not zero.lt(n_one)
    assert not zero.lt(zero)
    assert zero.lt(one)
    assert not one.lt(n_one)
    assert not one.lt(zero)
    assert not one.lt(one)


def test_vec_uxor():
    for k, v in UXOR.items():
        assert Vec[2](k).uxor() == Vec[1](v)


def test_eq():
    a = vec("4b0000")
    b = vec("4b1111")
    c = vec("4b0000")
    x = vec("4bXXXX")

    assert a.eq(c) == T
    assert a.eq(b) == F

    assert a.neq(b) == T
    assert a.neq(c) == F

    assert a.eq(x) == X
    assert x.eq(a) == X
    assert a.neq(x) == X
    assert x.neq(a) == X


def test_lt():
    a = int2vec(-1, 4)
    b = int2vec(0, 4)
    c = int2vec(1, 4)
    x = vec("4bXXXX")

    assert a.lt(b) == T
    assert b.lt(a) == F

    assert a.lt(c) == T
    assert c.lt(a) == F

    assert b.lt(c) == T
    assert c.lt(b) == F

    assert a.lt(x) == X
    assert x.lt(a) == X


def test_lte():
    a = int2vec(-1, 4)
    b = int2vec(0, 4)
    c = int2vec(1, 4)
    x = vec("4bXXXX")

    assert a.lte(b) == T
    assert b.lte(a) == F

    assert a.lte(c) == T
    assert c.lte(a) == F

    assert b.lte(c) == T
    assert c.lte(b) == F

    assert a.lte(x) == X
    assert x.lte(a) == X


def test_ltu():
    a = uint2vec(0, 4)
    b = uint2vec(1, 4)
    x = vec("4bXXXX")

    assert a.ltu(b) == T
    assert b.ltu(a) == F

    assert a.ltu(x) == X
    assert x.ltu(a) == X


def test_lteu():
    a = uint2vec(0, 4)
    b = uint2vec(1, 4)
    x = vec("4bXXXX")

    assert a.lteu(b) == T
    assert b.lteu(a) == F

    assert a.lteu(x) == X
    assert x.lteu(a) == X


def test_gt():
    a = int2vec(-1, 4)
    b = int2vec(0, 4)
    c = int2vec(1, 4)
    x = vec("4bXXXX")

    assert a.gt(b) == F
    assert b.gt(a) == T

    assert a.gt(c) == F
    assert c.gt(a) == T

    assert b.gt(c) == F
    assert c.gt(b) == T

    assert a.gt(x) == X
    assert x.gt(a) == X


def test_gte():
    a = int2vec(-1, 4)
    b = int2vec(0, 4)
    c = int2vec(1, 4)
    x = vec("4bXXXX")

    assert a.gte(b) == F
    assert b.gte(a) == T

    assert a.gte(c) == F
    assert c.gte(a) == T

    assert b.gte(c) == F
    assert c.gte(b) == T

    assert a.gte(x) == X
    assert x.gte(a) == X


def test_gtu():
    a = uint2vec(0, 4)
    b = uint2vec(1, 4)
    x = vec("4bXXXX")

    assert a.gtu(b) == F
    assert b.gtu(a) == T

    assert a.gtu(x) == X
    assert x.gtu(a) == X


def test_gteu():
    a = uint2vec(0, 4)
    b = uint2vec(1, 4)
    x = vec("4bXXXX")

    assert a.gteu(b) == F
    assert b.gteu(a) == T

    assert a.gteu(x) == X
    assert x.gteu(a) == X


def test_vec_match():
    a = vec("2b01")
    b = vec("1b-")
    c = vec("1bX")

    assert a._match("2b01")
    assert not a._match("2b00")
    assert not a._match("2b10")
    assert not a._match("2b11")

    assert a._match("2b0-")
    assert a._match("2b-1")
    assert a._match("2b--")
    assert not a._match("2b-0")
    assert not a._match("2b1-")
    assert not a._match("2bX1")
    assert not a._match("2b0X")

    assert a._match(vec("2b01"))
    assert not a._match(vec("2b00"))

    assert not b._match("1bX")
    assert b._match("1b0")
    assert b._match("1b1")
    assert b._match("1b-")

    assert not c._match("1b0")
    assert not c._match("1b1")
    assert not c._match("1b-")
    assert not c._match("1bX")

    with pytest.raises(TypeError):
        _ = a._match(42)  # pyright: ignore[reportArgumentType]


def test_vec_zext():
    v = Vec[4](0b10_01_10_01)
    with pytest.raises(ValueError):
        v.zext(-1)
    assert v.zext(0) is v
    assert v.zext(4) == Vec[8](0b01_01_01_01_10_01_10_01)


def test_vec_sext():
    v1 = Vec[4](0b10_01_10_01)
    v2 = Vec[4](0b01_10_01_10)
    with pytest.raises(ValueError):
        v1.sext(-1)
    assert v1.sext(0) is v1
    assert v1.sext(4) == Vec[8](0b10_10_10_10_10_01_10_01)
    assert v2.sext(0) is v2
    assert v2.sext(4) == Vec[8](0b01_01_01_01_01_10_01_10)


def test_vec_lsh():
    v = Vec[4](0b10_10_10_10)
    y, co = v.lsh(0)
    assert y is v and co == E
    assert v.lsh(1) == (Vec[4](0b10_10_10_01), Vec[1](0b10))
    assert v.lsh(2) == (Vec[4](0b10_10_01_01), Vec[2](0b10_10))
    assert v << 2 == Vec[4](0b10_10_01_01)
    assert v.lsh(3) == (Vec[4](0b10_01_01_01), Vec[3](0b10_10_10))
    assert v.lsh(4) == (Vec[4](0b01_01_01_01), Vec[4](0b10_10_10_10))

    with pytest.raises(ValueError):
        v.lsh(-1)
    with pytest.raises(ValueError):
        v.lsh(5)

    assert v.lsh(2, Vec[2](0b01_10)) == (Vec[4](0b10_10_01_10), Vec[2](0b10_10))
    with pytest.raises(ValueError):
        v.lsh(2, Vec[3](0b01_01_01))

    assert vec("2b01").lsh(vec("1bX")) == (vec("2bXX"), E)
    assert vec("2b01").lsh(vec("1b-")) == (vec("2b--"), E)
    assert vec("2b01").lsh(vec("1b1")) == (vec("2b10"), F)

    with pytest.raises(TypeError):
        v.lsh("foo")  # pyright: ignore[reportArgumentType]


def test_vec_rsh():
    v = Vec[4](0b10_10_10_10)
    y, co = v.rsh(0)
    assert y is v and co == E
    assert v.rsh(1) == (Vec[4](0b01_10_10_10), Vec[1](0b10))
    assert v.rsh(2) == (Vec[4](0b01_01_10_10), Vec[2](0b10_10))
    assert v >> 2 == Vec[4](0b01_01_10_10)
    assert v.rsh(3) == (Vec[4](0b01_01_01_10), Vec[3](0b10_10_10))
    assert v.rsh(4) == (Vec[4](0b01_01_01_01), Vec[4](0b10_10_10_10))

    with pytest.raises(ValueError):
        v.rsh(-1)
    with pytest.raises(ValueError):
        v.rsh(5)

    assert v.rsh(2, Vec[2](0b01_10)) == (Vec[4](0b01_10_10_10), Vec[2](0b10_10))
    with pytest.raises(ValueError):
        v.rsh(2, Vec[3](0b01_01_01))

    assert vec("2b01").rsh(vec("1bX")) == (vec("2bXX"), E)
    assert vec("2b01").rsh(vec("1b-")) == (vec("2b--"), E)
    assert vec("2b01").rsh(vec("1b1")) == (vec("2b00"), T)

    with pytest.raises(TypeError):
        v.rsh("foo")  # pyright: ignore[reportArgumentType]


def test_vec_arsh():
    v = Vec[4](0b10_10_10_10)
    assert v.arsh(0) == (Vec[4](0b10_10_10_10), E)
    assert v.arsh(1) == (Vec[4](0b10_10_10_10), Vec[1](0b10))
    assert v.arsh(2) == (Vec[4](0b10_10_10_10), Vec[2](0b10_10))
    assert v.arsh(3) == (Vec[4](0b10_10_10_10), Vec[3](0b10_10_10))
    assert v.arsh(4) == (Vec[4](0b10_10_10_10), Vec[4](0b10_10_10_10))

    v = Vec[4](0b01_10_10_10)
    assert v.arsh(0) == (Vec[4](0b01_10_10_10), E)
    assert v.arsh(1) == (Vec[4](0b01_01_10_10), Vec[1](0b10))
    assert v.arsh(2) == (Vec[4](0b01_01_01_10), Vec[2](0b10_10))
    assert v.arsh(3) == (Vec[4](0b01_01_01_01), Vec[3](0b10_10_10))
    assert v.arsh(4) == (Vec[4](0b01_01_01_01), Vec[4](0b01_10_10_10))

    with pytest.raises(ValueError):
        v.arsh(-1)
    with pytest.raises(ValueError):
        v.arsh(5)

    assert vec("2b01").arsh(vec("1bX")) == (vec("2bXX"), E)
    assert vec("2b01").arsh(vec("1b-")) == (vec("2b--"), E)
    assert vec("2b01").arsh(vec("1b1")) == (vec("2b00"), T)

    with pytest.raises(TypeError):
        v.arsh("foo")  # pyright: ignore[reportArgumentType]


ADD_VALS = [
    ("2b00", "2b00", F, "2b00", F, F),
    ("2b00", "2b01", F, "2b01", F, F),
    ("2b00", "2b10", F, "2b10", F, F),
    ("2b00", "2b11", F, "2b11", F, F),
    ("2b01", "2b00", F, "2b01", F, F),
    ("2b01", "2b01", F, "2b10", F, T),  # overflow
    ("2b01", "2b10", F, "2b11", F, F),
    ("2b01", "2b11", F, "2b00", T, F),
    ("2b10", "2b00", F, "2b10", F, F),
    ("2b10", "2b01", F, "2b11", F, F),
    ("2b10", "2b10", F, "2b00", T, T),  # overflow
    ("2b10", "2b11", F, "2b01", T, T),  # overflow
    ("2b11", "2b00", F, "2b11", F, F),
    ("2b11", "2b01", F, "2b00", T, F),
    ("2b11", "2b10", F, "2b01", T, T),  # overflow
    ("2b11", "2b11", F, "2b10", T, F),
    ("2b00", "2b00", T, "2b01", F, F),
    ("2b00", "2b01", T, "2b10", F, T),  # overflow
    ("2b00", "2b10", T, "2b11", F, F),
    ("2b00", "2b11", T, "2b00", T, F),
    ("2b01", "2b00", T, "2b10", F, T),  # overflow
    ("2b01", "2b01", T, "2b11", F, T),  # overflow
    ("2b01", "2b10", T, "2b00", T, F),
    ("2b01", "2b11", T, "2b01", T, F),
    ("2b10", "2b00", T, "2b11", F, F),
    ("2b10", "2b01", T, "2b00", T, F),
    ("2b10", "2b10", T, "2b01", T, T),  # overflow
    ("2b10", "2b11", T, "2b10", T, F),
    ("2b11", "2b00", T, "2b00", T, F),
    ("2b11", "2b01", T, "2b01", T, F),
    ("2b11", "2b10", T, "2b10", T, F),
    ("2b11", "2b11", T, "2b11", T, F),
]


def test_vec_add():
    """Test bits add method."""
    for a, b, ci, s, co, _ in ADD_VALS:
        a, b, s = vec(a), vec(b), vec(s)
        assert a.add(b, ci) == (s, co)


def test_vec_addsubnegops():
    """Test bits add/substract operators."""
    assert vec("2b00") + vec("2b00") == vec("2b00")
    assert vec("2b00") + vec("2b01") == vec("2b01")
    assert vec("2b01") + vec("2b00") == vec("2b01")
    assert vec("2b00") + vec("2b10") == vec("2b10")
    assert vec("2b01") + vec("2b01") == vec("2b10")
    assert vec("2b10") + vec("2b00") == vec("2b10")
    assert vec("2b00") + vec("2b11") == vec("2b11")
    assert vec("2b01") + vec("2b10") == vec("2b11")
    assert vec("2b10") + vec("2b01") == vec("2b11")
    assert vec("2b11") + vec("2b00") == vec("2b11")
    assert vec("2b01") + vec("2b11") == vec("2b00")
    assert vec("2b10") + vec("2b10") == vec("2b00")
    assert vec("2b11") + vec("2b01") == vec("2b00")
    assert vec("2b10") + vec("2b11") == vec("2b01")
    assert vec("2b11") + vec("2b10") == vec("2b01")
    assert vec("2b11") + vec("2b11") == vec("2b10")

    assert vec("2b00") - vec("2b11") == vec("2b01")
    assert vec("2b00") - vec("2b10") == vec("2b10")
    assert vec("2b01") - vec("2b11") == vec("2b10")
    assert vec("2b00") - vec("2b01") == vec("2b11")
    assert vec("2b01") - vec("2b10") == vec("2b11")
    assert vec("2b10") - vec("2b11") == vec("2b11")
    assert vec("2b00") - vec("2b00") == vec("2b00")
    assert vec("2b01") - vec("2b01") == vec("2b00")
    assert vec("2b10") - vec("2b10") == vec("2b00")
    assert vec("2b11") - vec("2b11") == vec("2b00")
    assert vec("2b01") - vec("2b00") == vec("2b01")
    assert vec("2b10") - vec("2b01") == vec("2b01")
    assert vec("2b11") - vec("2b10") == vec("2b01")
    assert vec("2b10") - vec("2b00") == vec("2b10")
    assert vec("2b11") - vec("2b01") == vec("2b10")
    assert vec("2b11") - vec("2b00") == vec("2b11")

    assert -vec("3b000") == vec("3b000")
    assert -vec("3b001") == vec("3b111")
    assert -vec("3b111") == vec("3b001")
    assert -vec("3b010") == vec("3b110")
    assert -vec("3b110") == vec("3b010")
    assert -vec("3b011") == vec("3b101")
    assert -vec("3b101") == vec("3b011")
    assert -vec("3b100") == vec("3b100")


def test_count():
    v = vec("8b-10X_-10X")
    assert v.count_xes() == 2
    assert v.count_zeros() == 2
    assert v.count_ones() == 2
    assert v.count_dcs() == 2

    assert vec("4b0000").count_ones() == 0
    assert vec("4b0001").count_ones() == 1
    assert vec("4b0011").count_ones() == 2
    assert vec("4b0111").count_ones() == 3
    assert vec("4b1111").count_ones() == 4

    assert not vec("4b0000").onehot()
    assert vec("4b1000").onehot()
    assert vec("4b0001").onehot()
    assert not vec("4b1001").onehot()
    assert not vec("4b1101").onehot()

    assert vec("4b0000").onehot0()
    assert vec("4b1000").onehot0()
    assert not vec("4b1010").onehot0()
    assert not vec("4b1011").onehot0()


UINT2VEC_VALS = {
    0: "",
    1: "1b1",
    2: "2b10",
    3: "2b11",
    4: "3b100",
    5: "3b101",
    6: "3b110",
    7: "3b111",
    8: "4b1000",
}

UINT2VEC_N_VALS = {
    (0, 0): "",
    (0, 1): "1b0",
    (0, 2): "2b00",
    (1, 1): "1b1",
    (1, 2): "2b01",
    (1, 3): "3b001",
    (1, 4): "4b0001",
    (2, 2): "2b10",
    (2, 3): "3b010",
    (2, 4): "4b0010",
    (3, 2): "2b11",
    (3, 3): "3b011",
    (3, 4): "4b0011",
    (4, 3): "3b100",
    (4, 4): "4b0100",
    (4, 5): "5b0_0100",
}


def test_uint2vec():
    # Negative inputs are invalid
    with pytest.raises(ValueError):
        uint2vec(-1)

    for i, s in UINT2VEC_VALS.items():
        v = uint2vec(i)
        assert str(v) == s
        assert v.to_uint() == i

    with pytest.raises(ValueError):
        uint2vec(1, 0)
    with pytest.raises(ValueError):
        uint2vec(2, 0)
    with pytest.raises(ValueError):
        uint2vec(2, 1)
    with pytest.raises(ValueError):
        uint2vec(3, 0)
    with pytest.raises(ValueError):
        uint2vec(3, 1)

    for (i, n), s in UINT2VEC_N_VALS.items():
        v = uint2vec(i, n)
        assert str(v) == s
        assert v.to_uint() == i


INT2VEC_VALS = {
    -8: "4b1000",
    -7: "4b1001",
    -6: "4b1010",
    -5: "4b1011",
    -4: "3b100",
    -3: "3b101",
    -2: "2b10",
    -1: "1b1",
    0: "1b0",
    1: "2b01",
    2: "3b010",
    3: "3b011",
    4: "4b0100",
    5: "4b0101",
    6: "4b0110",
    7: "4b0111",
    8: "5b0_1000",
}

INT2VEC_N_VALS = {
    (-5, 4): "4b1011",
    (-5, 5): "5b1_1011",
    (-5, 6): "6b11_1011",
    (-4, 3): "3b100",
    (-4, 4): "4b1100",
    (-4, 5): "5b1_1100",
    (-3, 3): "3b101",
    (-3, 4): "4b1101",
    (-3, 5): "5b1_1101",
    (-2, 2): "2b10",
    (-2, 3): "3b110",
    (-2, 4): "4b1110",
    (-1, 1): "1b1",
    (-1, 2): "2b11",
    (-1, 3): "3b111",
    (0, 1): "1b0",
    (0, 2): "2b00",
    (0, 3): "3b000",
    (1, 2): "2b01",
    (1, 3): "3b001",
    (1, 4): "4b0001",
    (2, 3): "3b010",
    (2, 4): "4b0010",
    (2, 5): "5b0_0010",
    (3, 3): "3b011",
    (3, 4): "4b0011",
    (3, 5): "5b0_0011",
    (4, 4): "4b0100",
    (4, 5): "5b0_0100",
    (4, 6): "6b00_0100",
}


def test_int2vec():
    for i, s in INT2VEC_VALS.items():
        v = int2vec(i)
        assert str(v) == s
        assert v.to_int() == i

    for (i, n), s in INT2VEC_N_VALS.items():
        v = int2vec(i, n)
        assert str(v) == s
        assert v.to_int() == i

    with pytest.raises(ValueError):
        int2vec(-5, 3)
    with pytest.raises(ValueError):
        int2vec(-4, 2)
    with pytest.raises(ValueError):
        int2vec(-3, 2)
    with pytest.raises(ValueError):
        int2vec(-2, 1)
    with pytest.raises(ValueError):
        int2vec(-1, 0)
    with pytest.raises(ValueError):
        int2vec(0, 0)
    with pytest.raises(ValueError):
        int2vec(1, 1)
    with pytest.raises(ValueError):
        int2vec(2, 2)
    with pytest.raises(ValueError):
        int2vec(3, 2)
    with pytest.raises(ValueError):
        int2vec(4, 3)


LIT2VEC_BIN = {
    "1b0": (1, 0b01),
    "1b1": (1, 0b10),
    "2b00": (2, 0b01_01),
    "2b01": (2, 0b01_10),
    "2b10": (2, 0b10_01),
    "2b11": (2, 0b10_10),
    "3b100": (3, 0b10_01_01),
    "3b101": (3, 0b10_01_10),
    "3b110": (3, 0b10_10_01),
    "3b111": (3, 0b10_10_10),
    "4b1000": (4, 0b10_01_01_01),
    "4b1001": (4, 0b10_01_01_10),
    "4b1010": (4, 0b10_01_10_01),
    "4b1011": (4, 0b10_01_10_10),
    "4b1100": (4, 0b10_10_01_01),
    "4b1101": (4, 0b10_10_01_10),
    "4b1110": (4, 0b10_10_10_01),
    "4b1111": (4, 0b10_10_10_10),
    "5b1_0000": (5, 0b10_01_01_01_01),
    "5b1_1111": (5, 0b10_10_10_10_10),
    "6b10_0000": (6, 0b10_01_01_01_01_01),
    "6b11_1111": (6, 0b10_10_10_10_10_10),
    "7b100_0000": (7, 0b10_01_01_01_01_01_01),
    "7b111_1111": (7, 0b10_10_10_10_10_10_10),
    "8b1000_0000": (8, 0b10_01_01_01_01_01_01_01),
    "8b1111_1111": (8, 0b10_10_10_10_10_10_10_10),
    "9b1_0000_0000": (9, 0b10_01_01_01_01_01_01_01_01),
    "9b1_1111_1111": (9, 0b10_10_10_10_10_10_10_10_10),
}


def test_lit2vec_bin():
    # Valid inputs
    for lit, (n, data) in LIT2VEC_BIN.items():
        v = vec(lit)
        assert len(v) == n and v.data == data

    v = vec("4b-1_0X")
    assert len(v) == 4 and v.data == 0b11_10_01_00
    v = vec("4bX01-")
    assert len(v) == 4 and v.data == 0b00_01_10_11

    # Not a literal
    with pytest.raises(ValueError):
        vec("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        vec("0b0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        vec("4b1XW0")

    # Size is too big
    with pytest.raises(ValueError):
        vec("8b1010")

    # Size is too small
    with pytest.raises(ValueError):
        vec("4b1010_1010")


LIT2VEC_HEX = {
    "1h0": (1, 0b01),
    "1h1": (1, 0b10),
    "2h0": (2, 0b01_01),
    "2h1": (2, 0b01_10),
    "2h2": (2, 0b10_01),
    "2h3": (2, 0b10_10),
    "3h4": (3, 0b10_01_01),
    "3h5": (3, 0b10_01_10),
    "3h6": (3, 0b10_10_01),
    "3h7": (3, 0b10_10_10),
    "4h8": (4, 0b10_01_01_01),
    "4h9": (4, 0b10_01_01_10),
    "4hA": (4, 0b10_01_10_01),
    "4hB": (4, 0b10_01_10_10),
    "4hC": (4, 0b10_10_01_01),
    "4hD": (4, 0b10_10_01_10),
    "4hE": (4, 0b10_10_10_01),
    "4hF": (4, 0b10_10_10_10),
    "5h10": (5, 0b10_01_01_01_01),
    "5h1F": (5, 0b10_10_10_10_10),
    "6h20": (6, 0b10_01_01_01_01_01),
    "6h3F": (6, 0b10_10_10_10_10_10),
    "7h40": (7, 0b10_01_01_01_01_01_01),
    "7h7F": (7, 0b10_10_10_10_10_10_10),
    "8h80": (8, 0b10_01_01_01_01_01_01_01),
    "8hFF": (8, 0b10_10_10_10_10_10_10_10),
    "9h100": (9, 0b10_01_01_01_01_01_01_01_01),
    "9h1FF": (9, 0b10_10_10_10_10_10_10_10_10),
}


def test_lit2vec_hex():
    # Valid inputs
    for lit, (n, data) in LIT2VEC_HEX.items():
        v = vec(lit)
        assert len(v) == n and v.data == data

    # Not a literal
    with pytest.raises(ValueError):
        vec("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        vec("0h0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        vec("8hd3@d_b33f")

    # Size is too big
    with pytest.raises(ValueError):
        vec("16hdead_beef")

    # Size is too small
    with pytest.raises(ValueError):
        vec("8hdead")

    # Invalid characters
    with pytest.raises(ValueError):
        vec("3h8")  # Only 0..7 is legal
    with pytest.raises(ValueError):
        vec("5h20")  # Only 0..1F is legal


def test_vec_basic():
    # n is non-negative
    with pytest.raises(ValueError):
        Vec[-1](42)

    # data in [0, 2**nbits)
    with pytest.raises(ValueError):
        Vec[4](-1)
    with pytest.raises(ValueError):
        Vec[4](2 ** (2 * 4))

    v = Vec[4](0b11_10_01_00)
    assert len(v) == 4

    assert v[3] == Vec[1](0b11)
    assert v[2] == Vec[1](0b10)
    assert v[1] == Vec[1](0b01)
    assert v[0] == Vec[1](0b00)

    assert v[0:1] == Vec[1](0b00)
    assert v[0:2] == Vec[2](0b01_00)
    assert v[0:3] == Vec[3](0b10_01_00)
    assert v[0:4] == Vec[4](0b11_10_01_00)
    assert v[1:2] == Vec[1](0b01)
    assert v[1:3] == Vec[2](0b10_01)
    assert v[1:4] == Vec[3](0b11_10_01)
    assert v[2:3] == Vec[1](0b10)
    assert v[2:4] == Vec[2](0b11_10)
    assert v[3:4] == Vec[1](0b11)

    with pytest.raises(TypeError):
        v["invalid"]  # pyright: ignore[reportArgumentType]

    assert list(v) == [Vec[1](0b00), Vec[1](0b01), Vec[1](0b10), Vec[1](0b11)]


def test_cat():
    v = vec("4b-10X")
    assert cat() == vec()
    assert cat(v) == v
    assert cat("2b0X", "2b-1") == vec("4b-10X")
    assert cat(vec("2b0X"), vec("2b-1")) == vec("4b-10X")
    assert cat(0, 1) == vec("2b10")

    with pytest.raises(TypeError):
        _ = cat(v, 42)


def test_rep():
    assert rep(vec("4b-10X"), 2) == vec("8b-10X_-10X")


def test_misc():
    assert str(xes(4)) == "4bXXXX"
    assert str(dcs(4)) == "4b----"

    with pytest.raises(ValueError):
        _ = xes(-4)
