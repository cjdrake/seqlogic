"""Test Logic Vector data type."""

# pylint: disable = pointless-statement

import pytest

from seqlogic.logic import logic
from seqlogic.logicvec import cat, nulls, ones, rep, uint2vec, vec, xes, zeros


def test_not():
    """Test logicvec NOT function."""
    x = vec([logic.N, logic.F, logic.T, logic.X])
    assert str(~x) == "vec(4bx01X)"


def test_nor():
    """Test logicvec NOR function."""
    x0 = vec("16bxxxx_1111_0000_XXXX")
    x1 = vec("16bx10X_x10X_x10X_x10X")
    assert str(x0.lnor(x1)) == "vec(16bx0xX_000X_x01X_XXXX)"


def test_or():
    """Test logicvec OR function."""
    x0 = vec("16bxxxx_1111_0000_XXXX")
    x1 = vec("16bx10X_x10X_x10X_x10X")
    assert str(x0 | x1) == "vec(16bx1xX_111X_x10X_XXXX)"


def test_nand():
    """Test logicvec NAND function."""
    x0 = vec("16bxxxx_1111_0000_XXXX")
    x1 = vec("16bx10X_x10X_x10X_x10X")
    assert str(x0.lnand(x1)) == "vec(16bxx1X_x01X_111X_XXXX)"


def test_and():
    """Test logicvec AND function."""
    x0 = vec("16bxxxx_1111_0000_XXXX")
    x1 = vec("16bx10X_x10X_x10X_x10X")
    assert str(x0 & x1) == "vec(16bxx0X_x10X_000X_XXXX)"


def test_xnor():
    """Test logicvec XNOR function."""
    x0 = vec("16bxxxx_1111_0000_XXXX")
    x1 = vec("16bx10X_x10X_x10X_x10X")
    assert str(x0.lxnor(x1)) == "vec(16bxxxX_x10X_x01X_XXXX)"


def test_xor():
    """Test logicvec XOR function."""
    x0 = vec("16bxxxx_1111_0000_XXXX")
    x1 = vec("16bx10X_x10X_x10X_x10X")
    assert str(x0 ^ x1) == "vec(16bxxxX_x01X_x10X_XXXX)"


def test_uor():
    """Test logicvec unary OR method."""
    assert vec("2bXX").ulor() is logic.N
    assert vec("2b0X").ulor() is logic.N
    assert vec("2b1X").ulor() is logic.N
    assert vec("2bxX").ulor() is logic.N
    assert vec("2bX0").ulor() is logic.N
    assert vec("2b00").ulor() is logic.F
    assert vec("2b10").ulor() is logic.T
    assert vec("2bx0").ulor() is logic.X
    assert vec("2bX1").ulor() is logic.N
    assert vec("2b01").ulor() is logic.T
    assert vec("2b11").ulor() is logic.T
    assert vec("2bx1").ulor() is logic.T
    assert vec("2bXx").ulor() is logic.N
    assert vec("2b0x").ulor() is logic.X
    assert vec("2b1x").ulor() is logic.T
    assert vec("2bxx").ulor() is logic.X


def test_uand():
    """Test logicvec unary and method."""
    assert vec("2bXX").uland() is logic.N
    assert vec("2b0X").uland() is logic.N
    assert vec("2b1X").uland() is logic.N
    assert vec("2bxX").uland() is logic.N
    assert vec("2bX0").uland() is logic.N
    assert vec("2b00").uland() is logic.F
    assert vec("2b10").uland() is logic.F
    assert vec("2bx0").uland() is logic.F
    assert vec("2bX1").uland() is logic.N
    assert vec("2b01").uland() is logic.F
    assert vec("2b11").uland() is logic.T
    assert vec("2bx1").uland() is logic.X
    assert vec("2bXx").uland() is logic.N
    assert vec("2b0x").uland() is logic.F
    assert vec("2b1x").uland() is logic.X
    assert vec("2bxx").uland() is logic.X


def test_uxor():
    """Test logicvec unary xor method."""
    assert vec("2bXX").ulxor() is logic.N
    assert vec("2b0X").ulxor() is logic.N
    assert vec("2b1X").ulxor() is logic.N
    assert vec("2bxX").ulxor() is logic.N
    assert vec("2bX0").ulxor() is logic.N
    assert vec("2b00").ulxor() is logic.F
    assert vec("2b10").ulxor() is logic.T
    assert vec("2bx0").ulxor() is logic.X
    assert vec("2bX1").ulxor() is logic.N
    assert vec("2b01").ulxor() is logic.T
    assert vec("2b11").ulxor() is logic.F
    assert vec("2bx1").ulxor() is logic.X
    assert vec("2bXx").ulxor() is logic.N
    assert vec("2b0x").ulxor() is logic.X
    assert vec("2b1x").ulxor() is logic.X
    assert vec("2bxx").ulxor() is logic.X


def test_zext():
    """Test logicvec zext method."""
    assert vec("4b1010").zext(4) == vec("8b0000_1010")
    # Zero extension on multi-dimensional array will flatten
    assert vec(["4b0000", "4b1111"]).zext(2) == vec("10b00_1111_0000")


def test_sext():
    """Test logicvec sext method."""
    assert vec("4b1010").sext(4) == vec("8b1111_1010")
    assert vec("4b0101").sext(4) == vec("8b0000_0101")
    # Sign extension of multi-dimensional array will flatten
    assert vec(["4b0000", "4b1111"]).sext(2) == vec("10b11_1111_0000")


def test_lsh():
    """Test logicvec lsh method."""
    v = vec("4b1111")
    assert v.lsh(0) == (vec("4b1111"), vec())
    assert v.lsh(1) == (vec("4b1110"), vec("1b1"))
    assert v.lsh(2) == (vec("4b1100"), vec("2b11"))
    assert v << 2 == vec("4b1100")
    assert v.lsh(3) == (vec("4b1000"), vec("3b111"))
    assert v.lsh(4) == (vec("4b0000"), vec("4b1111"))
    with pytest.raises(ValueError):
        v.lsh(-1)
    with pytest.raises(ValueError):
        v.lsh(5)

    assert v.lsh(2, vec("2b00")) == (vec("4b1100"), vec("2b11"))
    with pytest.raises(ValueError):
        assert v.lsh(2, vec("3b000"))

    assert vec(["4b0000", "4b1111"]).lsh(2) == (vec("8b1100_0000"), vec("2b11"))


def test_rsh():
    """Test logicvec rsh method."""
    v = vec("4b1111")
    assert v.rsh(0) == (vec("4b1111"), vec())
    assert v.rsh(1) == (vec("4b0111"), vec("1b1"))
    assert v.rsh(2) == (vec("4b0011"), vec("2b11"))
    assert v >> 2 == vec("4b0011")
    assert v.rsh(3) == (vec("4b0001"), vec("3b111"))
    assert v.rsh(4) == (vec("4b0000"), vec("4b1111"))
    with pytest.raises(ValueError):
        v.rsh(-1)
    with pytest.raises(ValueError):
        v.rsh(5)

    assert v.rsh(2, vec("2b00")) == (vec("4b0011"), vec("2b11"))
    with pytest.raises(ValueError):
        assert v.rsh(2, vec("3b000"))

    assert vec(["4b0000", "4b1111"]).rsh(2) == (vec("8b0011_1100"), vec("2b00"))


def test_arsh():
    """Test logicvec arsh method."""
    v = vec("4b1111")
    assert v.arsh(0) == (vec("4b1111"), vec())
    assert v.arsh(1) == (vec("4b1111"), vec("1b1"))
    assert v.arsh(2) == (vec("4b1111"), vec("2b11"))
    assert v.arsh(3) == (vec("4b1111"), vec("3b111"))
    assert v.arsh(4) == (vec("4b1111"), vec("4b1111"))

    v = vec("4b0111")
    assert v.arsh(0) == (vec("4b0111"), vec())
    assert v.arsh(1) == (vec("4b0011"), vec("1b1"))
    assert v.arsh(2) == (vec("4b0001"), vec("2b11"))
    assert v.arsh(3) == (vec("4b0000"), vec("3b111"))
    assert v.arsh(4) == (vec("4b0000"), vec("4b0111"))

    with pytest.raises(ValueError):
        v.arsh(-1)
    with pytest.raises(ValueError):
        v.arsh(5)

    assert vec(["4b0000", "4b1111"]).arsh(2) == (vec("8b1111_1100"), vec("2b00"))


ADD_VALS = [
    ("2b00", "2b00", logic.F, "2b00", logic.F, logic.F),
    ("2b00", "2b01", logic.F, "2b01", logic.F, logic.F),
    ("2b00", "2b10", logic.F, "2b10", logic.F, logic.F),
    ("2b00", "2b11", logic.F, "2b11", logic.F, logic.F),
    ("2b01", "2b00", logic.F, "2b01", logic.F, logic.F),
    ("2b01", "2b01", logic.F, "2b10", logic.F, logic.T),  # overflow
    ("2b01", "2b10", logic.F, "2b11", logic.F, logic.F),
    ("2b01", "2b11", logic.F, "2b00", logic.T, logic.F),
    ("2b10", "2b00", logic.F, "2b10", logic.F, logic.F),
    ("2b10", "2b01", logic.F, "2b11", logic.F, logic.F),
    ("2b10", "2b10", logic.F, "2b00", logic.T, logic.T),  # overflow
    ("2b10", "2b11", logic.F, "2b01", logic.T, logic.T),  # overflow
    ("2b11", "2b00", logic.F, "2b11", logic.F, logic.F),
    ("2b11", "2b01", logic.F, "2b00", logic.T, logic.F),
    ("2b11", "2b10", logic.F, "2b01", logic.T, logic.T),  # overflow
    ("2b11", "2b11", logic.F, "2b10", logic.T, logic.F),
    ("2b00", "2b00", logic.T, "2b01", logic.F, logic.F),
    ("2b00", "2b01", logic.T, "2b10", logic.F, logic.T),  # overflow
    ("2b00", "2b10", logic.T, "2b11", logic.F, logic.F),
    ("2b00", "2b11", logic.T, "2b00", logic.T, logic.F),
    ("2b01", "2b00", logic.T, "2b10", logic.F, logic.T),  # overflow
    ("2b01", "2b01", logic.T, "2b11", logic.F, logic.T),  # overflow
    ("2b01", "2b10", logic.T, "2b00", logic.T, logic.F),
    ("2b01", "2b11", logic.T, "2b01", logic.T, logic.F),
    ("2b10", "2b00", logic.T, "2b11", logic.F, logic.F),
    ("2b10", "2b01", logic.T, "2b00", logic.T, logic.F),
    ("2b10", "2b10", logic.T, "2b01", logic.T, logic.T),  # overflow
    ("2b10", "2b11", logic.T, "2b10", logic.T, logic.F),
    ("2b11", "2b00", logic.T, "2b00", logic.T, logic.F),
    ("2b11", "2b01", logic.T, "2b01", logic.T, logic.F),
    ("2b11", "2b10", logic.T, "2b10", logic.T, logic.F),
    ("2b11", "2b11", logic.T, "2b11", logic.T, logic.F),
]


def test_add():
    """Test logicvec add method."""
    for a, b, ci, s, co, v in ADD_VALS:
        a, b, s = vec(a), vec(b), vec(s)
        assert a.add(b, ci) == (s, co, v)


def test_addsubops():
    """Test logicvec add/substract operators."""
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


def test_operand_shape_mismatch():
    """Test vector operations with mismatching shapes.

    We could implement something like Verilog's loose typing, but for the time
    being just treat this as illegal.
    """
    x0 = vec("4b1010")
    x1 = vec("8b0101_0101")
    with pytest.raises(ValueError):
        x0.lnor(x1)
    with pytest.raises(ValueError):
        x0 | x1
    with pytest.raises(ValueError):
        x0.lnand(x1)
    with pytest.raises(ValueError):
        x0 & x1
    with pytest.raises(ValueError):
        x0.lxnor(x1)
    with pytest.raises(ValueError):
        x0 ^ x1


def test_parse_str_literal():
    """Test parsing of vector string literals."""
    # literal doesn't match size
    with pytest.raises(ValueError):
        vec("4b1010_1010")
    with pytest.raises(ValueError):
        vec("8b1010")
    with pytest.raises(ValueError):
        vec("16hdead_beef")
    with pytest.raises(ValueError):
        vec("8hdead")

    # Invalid input
    with pytest.raises(ValueError):
        vec("invalid")

    # Valid input
    v = vec("4bx1_0X")
    assert v.pcs.bits == 0b11_10_01_00
    v = vec("64hFeDc_Ba98_7654_3210")
    assert v.pcs.bits == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655
    v = vec("64hfEdC_bA98_7654_3210")
    assert v.pcs.bits == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655


def test_uint2vec():
    """Test parsing int literals."""
    with pytest.raises(ValueError):
        uint2vec(-1)

    assert str(uint2vec(0)) == "vec([0])"
    assert str(uint2vec(1)) == "vec([1])"
    assert str(uint2vec(2)) == "vec(2b10)"
    assert str(uint2vec(3)) == "vec(2b11)"
    assert str(uint2vec(4)) == "vec(3b100)"
    assert str(uint2vec(5)) == "vec(3b101)"
    assert str(uint2vec(6)) == "vec(3b110)"
    assert str(uint2vec(7)) == "vec(3b111)"
    assert str(uint2vec(8)) == "vec(4b1000)"

    assert str(uint2vec(0, size=4)) == "vec(4b0000)"
    assert str(uint2vec(1, size=4)) == "vec(4b0001)"
    assert str(uint2vec(2, size=4)) == "vec(4b0010)"
    assert str(uint2vec(3, size=4)) == "vec(4b0011)"
    assert str(uint2vec(4, size=4)) == "vec(4b0100)"
    assert str(uint2vec(5, size=4)) == "vec(4b0101)"
    assert str(uint2vec(6, size=4)) == "vec(4b0110)"
    assert str(uint2vec(7, size=4)) == "vec(4b0111)"
    assert str(uint2vec(8, size=4)) == "vec(4b1000)"

    with pytest.raises(ValueError):
        uint2vec(8, size=3)


def test_empty():
    """Test empty vector."""
    v = vec()

    # Test properties
    assert v.shape == (0,)
    assert v.pcs.bits == 0
    assert v.ndim == 1
    assert v.size == 0
    assert v.pcs.nbits == 0
    assert list(v.flat) == []  # pylint: disable = use-implicit-booleaness-not-comparison

    assert v.flatten() == v

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([])"

    # Test __len__
    assert len(v) == 0

    # Test __iter__
    assert list(v) == []  # pylint: disable = use-implicit-booleaness-not-comparison

    # Test __getitem__
    with pytest.raises(IndexError):
        _ = v[0]

    # Test __eq__
    assert v == vec()
    assert v == v.reshape((0,))
    assert v != vec(0)

    # Test to_uint, to_int
    assert v.to_uint() == 0
    assert v.to_int() == 0


def test_scalar():
    """Test scalar (vector w/ one element)."""
    vn = vec(logic.N)
    v0 = vec(0)
    v1 = vec(1)
    vx = vec(logic.X)

    # Test properties
    assert v0.shape == (1,)
    assert v0.pcs.bits == logic.F.value
    assert v0.ndim == 1
    assert v0.size == 1
    assert v0.pcs.nbits == 2
    assert list(v0.flat) == [logic.F]

    assert v0.flatten() == v0

    # Test __str__ and __repr__
    assert str(vn) == repr(vn) == "vec([X])"
    assert str(v0) == repr(v0) == "vec([0])"
    assert str(v1) == repr(v1) == "vec([1])"
    assert str(vx) == repr(vx) == "vec([x])"

    # Test __len__
    assert len(v0) == 1

    # Test __iter__
    assert list(v0) == [logic.F]

    # Test __getitem__
    assert v0[0] is logic.F

    # Test __eq__
    assert v0 == vec(0)
    assert v0 == v0.reshape((1,))
    assert v0 != vec()
    assert v0 != vec(1)

    # Test to_uint, to_int
    with pytest.raises(ValueError):
        assert vn.to_uint()
    assert v0.to_uint() == 0
    assert v0.to_int() == 0
    assert v1.to_uint() == 1
    assert v1.to_int() == -1
    with pytest.raises(ValueError):
        assert vx.to_uint()


def test_rank1_str():
    """Test vec rank1 string input."""
    v = vec("8bx10X_x10X")
    data = 0b11100100_11100100
    xs = [
        logic.N,
        logic.F,
        logic.T,
        logic.X,
    ] * 2

    # Test properties
    assert v.shape == (8,)
    assert v.pcs.bits == data
    assert v.ndim == 1
    assert v.size == 8
    assert v.pcs.nbits == 16
    assert list(v.flat) == xs

    assert v.flatten() == v

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec(8bx10X_x10X)"

    # Test __len__
    assert len(v) == 8

    # Test __iter__
    assert list(v) == xs

    # Test __getitem__
    assert v[0] is logic.N
    assert v[1] is logic.F
    assert v[6] is logic.T
    assert v[7] is logic.X

    # Test __eq__
    assert v == vec("8bx10X_x10X")
    # Same data, different shape
    assert v != v.reshape((2, 4))
    with pytest.raises(ValueError):
        v.reshape((42,))
    # Different data, same shape
    assert v != vec("8b0000_0000")

    # Test to_uint
    assert vec("16b1101_1110_1010_1101").to_uint() == 0xDEAD

    # Test to_int
    assert vec("4b0000").to_int() == 0
    assert vec("4b0001").to_int() == 1
    assert vec("4b0110").to_int() == 6
    assert vec("4b0111").to_int() == 7
    assert vec("4b1000").to_int() == -8
    assert vec("4b1001").to_int() == -7
    assert vec("4b1110").to_int() == -2
    assert vec("4b1111").to_int() == -1


def test_rank1_logic():
    """Test vec function w/ rank1 logic input."""
    xs = [logic.N, logic.F, logic.T, logic.X]
    v1 = vec(xs)
    v2 = vec([0, 1, 0, 1])
    with pytest.raises(TypeError):
        _ = vec([0, "invalid"])

    # Test properties
    assert v1.shape == (4,)
    assert v1.pcs.bits == 0b11100100
    assert v1.ndim == 1
    assert v1.size == 4
    assert v1.pcs.nbits == 8
    assert list(v1.flat) == xs

    # Test __str__ and __repr__
    assert str(v1) == repr(v1) == "vec(4bx10X)"
    assert str(v2) == repr(v2) == "vec(4b1010)"


def test_rank2_str():
    """Test vec function w/ rank2 str input."""
    v = vec(["4bx10X", "4bx10X"])

    assert v.flatten() == vec("8bx10X_x10X")

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([4bx10X, 4bx10X])"


def test_rank2_vec():
    """Test vec function w/ rank2 vec input."""
    v = vec([vec("4bx10X"), vec("4bx10X")])

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([4bx10X, 4bx10X])"


def test_rank2_errors():
    """Test vec function rank2 errors."""
    # Mismatched str literal
    with pytest.raises(TypeError):
        vec(["4bx10X", "3b10X"])
    # logicvec followed by some invalid type
    with pytest.raises(TypeError):
        vec(["4bx10X", 42])


R3VEC = """\
vec([[4bx10X, 4bx10X],
     [4bx10X, 4bx10X]])"""


def test_rank3_vec():
    """Test vec function w/ rank3 input."""
    v = vec(
        [
            [vec("4bx10X"), vec("4bx10X")],
            [vec("4bx10X"), vec("4bx10X")],
        ]
    )

    assert v.flatten() == vec("16bx10X_x10X_x10X_x10X")

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R3VEC


R4VEC = """\
vec([[[4bx10X, 4bx10X],
      [4bx10X, 4bx10X]],

     [[4bx10X, 4bx10X],
      [4bx10X, 4bx10X]]])"""


def test_rank4_vec():
    """Test vec function w/ rank4 input."""
    v = vec(
        [
            [[vec("4bx10X"), vec("4bx10X")], [vec("4bx10X"), vec("4bx10X")]],
            [[vec("4bx10X"), vec("4bx10X")], [vec("4bx10X"), vec("4bx10X")]],
        ]
    )

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R4VEC


def test_invalid_vec():
    """Test vec function invalid input."""
    with pytest.raises(TypeError):
        vec(42)


def test_cat():
    """Test logicvec cat function."""
    assert cat([]) == vec()
    assert cat([False, True, False, True]) == vec("4b1010")

    with pytest.raises(TypeError):
        cat(["invalid"])

    v = cat([vec("2b00"), vec("2b01"), vec("2b10"), vec("2b11")], flatten=True)
    assert v == vec("8b11100100")
    assert v.shape == (8,)

    v = cat([vec("2b00"), vec("2b01"), vec("2b10"), vec("2b11")], flatten=False)
    assert v.shape == (4, 2)

    v = cat([vec("1b0"), vec("2b01"), vec("1b1"), vec("2b11")], flatten=True)
    assert v.shape == (6,)

    v = cat([vec("1b0"), vec("2b01"), vec("1b1"), vec("2b11")], flatten=False)
    assert v.shape == (6,)

    # Incompatible shapes
    with pytest.raises(ValueError):
        cat([vec("2b00"), vec([vec("2b00"), vec("2b00")])])


def test_rep():
    """Test logicvec rep function."""
    v = rep(vec("2b00"), 4, flatten=True)
    assert v == vec("8b0000_0000")
    assert v.shape == (8,)

    v = rep(vec("2b00"), 4, flatten=False)
    assert v.shape == (4, 2)


def test_consts():
    """Test logicvec constants."""
    assert nulls((8,)) == vec("8bXXXX_XXXX")
    assert zeros((8,)) == vec("8b0000_0000")
    assert ones((8,)) == vec("8b1111_1111")
    assert xes((8,)) == vec("8bxxxx_xxxx")


def test_slicing():
    """Test logicvec slicing behavior."""
    v = vec(
        [
            [vec("4b0000"), vec("4b0001"), vec("4b0010"), vec("4b0011")],
            [vec("4b0100"), vec("4b0101"), vec("4b0110"), vec("4b0111")],
            [vec("4b1000"), vec("4b1001"), vec("4b1010"), vec("4b1011")],
            [vec("4b1100"), vec("4b1101"), vec("4b1110"), vec("4b1111")],
        ]
    )

    assert v.shape == (4, 4, 4)

    with pytest.raises(IndexError):
        v[-5]
    with pytest.raises(TypeError):
        v["invalid"]

    assert v == v[:]
    assert v == v[0:4]
    assert v == v[-4:]
    assert v == v[-5:]
    assert v == v[:, :]
    assert v == v[:, :, :]

    assert v[0] == v[0, :]
    assert v[0] == v[0, 0:4]
    assert v[0] == v[0, -4:]
    assert v[0] == v[0, -5:]
    assert v[0] == v[0, :, :]

    assert v[0] == vec([vec("4b0000"), vec("4b0001"), vec("4b0010"), vec("4b0011")])
    assert v[1] == vec([vec("4b0100"), vec("4b0101"), vec("4b0110"), vec("4b0111")])
    assert v[2] == vec([vec("4b1000"), vec("4b1001"), vec("4b1010"), vec("4b1011")])
    assert v[3] == vec([vec("4b1100"), vec("4b1101"), vec("4b1110"), vec("4b1111")])

    assert v[0, 0] == v[0, 0, :]
    assert v[0, 0] == v[0, 0, 0:4]
    assert v[0, 0] == v[0, 0, -4:]
    assert v[0, 0] == v[0, 0, -5:]

    assert v[0, 0] == vec("4b0000")
    assert v[1, 1] == vec("4b0101")
    assert v[2, 2] == vec("4b1010")
    assert v[3, 3] == vec("4b1111")

    assert v[0, :, 0] == vec("4b1010")
    assert v[1, :, 1] == vec("4b1100")
    assert v[2, :, 2] == vec("4b0000")
    assert v[3, :, 3] == vec("4b1111")

    assert v[0, 0, :-1] == vec("3b000")
    assert v[0, 0, :-2] == vec("2b00")
    assert v[0, 0, :-3] == vec("1b0")
    assert v[0, 0, :-4] == vec()

    assert v[0, 0, 0] == logic.F
    assert v[0, vec("2b00"), 0] == logic.F
    assert v[-4, -4, -4] == logic.F
    assert v[3, 3, 3] == logic.T
    assert v[3, vec("2b11"), 3] == logic.T
    assert v[-1, -1, -1] == logic.T

    with pytest.raises(ValueError):
        v[0, 0, 0, 0]
    with pytest.raises(TypeError):
        v["invalid"]


def test_countbits():
    """Test logicvec countbits methods."""
    v = vec("8bx10X_x10X")
    assert v.countbits({logic.F, logic.T}) == 4
    assert v.countbits({logic.N, logic.X}) == 4

    assert vec("4b0000").countones() == 0
    assert vec("4b0001").countones() == 1
    assert vec("4b0011").countones() == 2
    assert vec("4b0111").countones() == 3
    assert vec("4b1111").countones() == 4

    assert not vec("4b0000").onehot()
    assert vec("4b1000").onehot()
    assert vec("4b0001").onehot()
    assert not vec("4b1001").onehot()
    assert not vec("4b1101").onehot()

    assert vec("4b0000").onehot0()
    assert vec("4b1000").onehot0()
    assert not vec("4b1010").onehot0()
    assert not vec("4b1011").onehot0()

    assert not vec("4b0000").isunknown()
    assert not vec("4b1111").isunknown()
    assert vec("4b0x01").isunknown()
    assert vec("4b01X1").isunknown()
