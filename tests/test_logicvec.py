"""
Test Logic Vector Data Type
"""

# pylint: disable = pointless-statement

import pytest

from seqlogic.logic import logic
from seqlogic.logicvec import cat, nulls, ones, rep, uint2vec, vec, xes, zeros


def test_not():
    """Test logic_vector NOT function"""
    x = vec([logic.N, logic.F, logic.T, logic.X])
    assert str(~x) == "vec(4'bx01X)"


def test_nor():
    """Test logic_vector NOR function"""
    x0 = vec("16'bxxxx_1111_0000_XXXX")
    x1 = vec("16'bx10X_x10X_x10X_x10X")
    assert str(x0.nor(x1)) == "vec(16'bx0xX_000X_x01X_XXXX)"


def test_or():
    """Test logic_vector OR function"""
    x0 = vec("16'bxxxx_1111_0000_XXXX")
    x1 = vec("16'bx10X_x10X_x10X_x10X")
    assert str(x0 | x1) == "vec(16'bx1xX_111X_x10X_XXXX)"


def test_nand():
    """Test logic_vector NAND function"""
    x0 = vec("16'bxxxx_1111_0000_XXXX")
    x1 = vec("16'bx10X_x10X_x10X_x10X")
    assert str(x0.nand(x1)) == "vec(16'bxx1X_x01X_111X_XXXX)"


def test_and():
    """Test logic_vector AND function"""
    x0 = vec("16'bxxxx_1111_0000_XXXX")
    x1 = vec("16'bx10X_x10X_x10X_x10X")
    assert str(x0 & x1) == "vec(16'bxx0X_x10X_000X_XXXX)"


def test_xnor():
    """Test logic_vector XNOR function"""
    x0 = vec("16'bxxxx_1111_0000_XXXX")
    x1 = vec("16'bx10X_x10X_x10X_x10X")
    assert str(x0.xnor(x1)) == "vec(16'bxxxX_x10X_x01X_XXXX)"


def test_xor():
    """Test logic_vector XOR function"""
    x0 = vec("16'bxxxx_1111_0000_XXXX")
    x1 = vec("16'bx10X_x10X_x10X_x10X")
    assert str(x0 ^ x1) == "vec(16'bxxxX_x01X_x10X_XXXX)"


def test_uor():
    assert vec("2'bXX").uor() is logic.N
    assert vec("2'b0X").uor() is logic.N
    assert vec("2'b1X").uor() is logic.N
    assert vec("2'bxX").uor() is logic.N
    assert vec("2'bX0").uor() is logic.N
    assert vec("2'b00").uor() is logic.F
    assert vec("2'b10").uor() is logic.T
    assert vec("2'bx0").uor() is logic.X
    assert vec("2'bX1").uor() is logic.N
    assert vec("2'b01").uor() is logic.T
    assert vec("2'b11").uor() is logic.T
    assert vec("2'bx1").uor() is logic.T
    assert vec("2'bXx").uor() is logic.N
    assert vec("2'b0x").uor() is logic.X
    assert vec("2'b1x").uor() is logic.T
    assert vec("2'bxx").uor() is logic.X


def test_uand():
    assert vec("2'bXX").uand() is logic.N
    assert vec("2'b0X").uand() is logic.N
    assert vec("2'b1X").uand() is logic.N
    assert vec("2'bxX").uand() is logic.N
    assert vec("2'bX0").uand() is logic.N
    assert vec("2'b00").uand() is logic.F
    assert vec("2'b10").uand() is logic.F
    assert vec("2'bx0").uand() is logic.F
    assert vec("2'bX1").uand() is logic.N
    assert vec("2'b01").uand() is logic.F
    assert vec("2'b11").uand() is logic.T
    assert vec("2'bx1").uand() is logic.X
    assert vec("2'bXx").uand() is logic.N
    assert vec("2'b0x").uand() is logic.F
    assert vec("2'b1x").uand() is logic.X
    assert vec("2'bxx").uand() is logic.X


def test_uxor():
    assert vec("2'bXX").uxor() is logic.N
    assert vec("2'b0X").uxor() is logic.N
    assert vec("2'b1X").uxor() is logic.N
    assert vec("2'bxX").uxor() is logic.N
    assert vec("2'bX0").uxor() is logic.N
    assert vec("2'b00").uxor() is logic.F
    assert vec("2'b10").uxor() is logic.T
    assert vec("2'bx0").uxor() is logic.X
    assert vec("2'bX1").uxor() is logic.N
    assert vec("2'b01").uxor() is logic.T
    assert vec("2'b11").uxor() is logic.F
    assert vec("2'bx1").uxor() is logic.X
    assert vec("2'bXx").uxor() is logic.N
    assert vec("2'b0x").uxor() is logic.X
    assert vec("2'b1x").uxor() is logic.X
    assert vec("2'bxx").uxor() is logic.X


def test_zext():
    v = vec("4'b1010")
    assert v.zext(4) == vec("8'b0000_1010")

    with pytest.raises(ValueError):
        vec(["4'b0000", "4'b1111"]).zext(2)


def test_lsh():
    v = vec("4'b1111")
    assert v.lsh(0) == (vec("4'b1111"), vec())
    assert v.lsh(1) == (vec("4'b1110"), vec("1'b1"))
    assert v.lsh(2) == (vec("4'b1100"), vec("2'b11"))
    assert v << 2 == vec("4'b1100")
    assert v.lsh(3) == (vec("4'b1000"), vec("3'b111"))
    assert v.lsh(4) == (vec("4'b0000"), vec("4'b1111"))
    with pytest.raises(ValueError):
        v.lsh(-1)
    with pytest.raises(ValueError):
        v.lsh(5)

    assert v.lsh(2, vec("2'b00")) == (vec("4'b1100"), vec("2'b11"))
    with pytest.raises(ValueError):
        assert v.lsh(2, vec("3'b000"))

    with pytest.raises(ValueError):
        vec(["4'b0000", "4'b1111"]).lsh(2)


def test_rsh():
    v = vec("4'b1111")
    assert v.rsh(0) == (vec("4'b1111"), vec())
    assert v.rsh(1) == (vec("4'b0111"), vec("1'b1"))
    assert v.rsh(2) == (vec("4'b0011"), vec("2'b11"))
    assert v >> 2 == vec("4'b0011")
    assert v.rsh(3) == (vec("4'b0001"), vec("3'b111"))
    assert v.rsh(4) == (vec("4'b0000"), vec("4'b1111"))
    with pytest.raises(ValueError):
        v.rsh(-1)
    with pytest.raises(ValueError):
        v.rsh(5)

    assert v.rsh(2, vec("2'b00")) == (vec("4'b0011"), vec("2'b11"))
    with pytest.raises(ValueError):
        assert v.rsh(2, vec("3'b000"))

    with pytest.raises(ValueError):
        vec(["4'b0000", "4'b1111"]).rsh(2)


def test_arsh():
    v = vec("4'b1111")
    assert v.arsh(0) == (vec("4'b1111"), vec())
    assert v.arsh(1) == (vec("4'b1111"), vec("1'b1"))
    assert v.arsh(2) == (vec("4'b1111"), vec("2'b11"))
    assert v.arsh(3) == (vec("4'b1111"), vec("3'b111"))
    assert v.arsh(4) == (vec("4'b1111"), vec("4'b1111"))

    v = vec("4'b0111")
    assert v.arsh(0) == (vec("4'b0111"), vec())
    assert v.arsh(1) == (vec("4'b0011"), vec("1'b1"))
    assert v.arsh(2) == (vec("4'b0001"), vec("2'b11"))
    assert v.arsh(3) == (vec("4'b0000"), vec("3'b111"))
    assert v.arsh(4) == (vec("4'b0000"), vec("4'b0111"))

    with pytest.raises(ValueError):
        v.arsh(-1)
    with pytest.raises(ValueError):
        v.arsh(5)

    with pytest.raises(ValueError):
        vec(["4'b0000", "4'b1111"]).arsh(2)


def test_operand_shape_mismatch():
    """Test vector operations with mismatching shapes.

    We could implement something like Verilog's loose typing, but for the time
    being just treat this as illegal.
    """
    x0 = vec("4'b1010")
    x1 = vec("8'b0101_0101")
    with pytest.raises(ValueError):
        x0.nor(x1)
    with pytest.raises(ValueError):
        x0 | x1
    with pytest.raises(ValueError):
        x0.nand(x1)
    with pytest.raises(ValueError):
        x0 & x1
    with pytest.raises(ValueError):
        x0.xnor(x1)
    with pytest.raises(ValueError):
        x0 ^ x1


def test_parse_str_literal():
    """Test parsing of vector string literals."""

    # literal doesn't match size
    with pytest.raises(ValueError):
        vec("4'b1010_1010")
    with pytest.raises(ValueError):
        vec("8'b1010")

    # Invalid input
    with pytest.raises(ValueError):
        vec("invalid")

    # Valid input
    v = vec("4'bx1_0X")
    assert v.data == 0b11_10_01_00


def test_uint2vec():
    """Test parsing int literals."""
    with pytest.raises(ValueError):
        uint2vec(-1)

    assert str(uint2vec(0)) == "vec([0])"
    assert str(uint2vec(1)) == "vec([1])"
    assert str(uint2vec(2)) == "vec(2'b10)"
    assert str(uint2vec(3)) == "vec(2'b11)"
    assert str(uint2vec(4)) == "vec(3'b100)"
    assert str(uint2vec(5)) == "vec(3'b101)"
    assert str(uint2vec(6)) == "vec(3'b110)"
    assert str(uint2vec(7)) == "vec(3'b111)"
    assert str(uint2vec(8)) == "vec(4'b1000)"

    assert str(uint2vec(0, size=4)) == "vec(4'b0000)"
    assert str(uint2vec(1, size=4)) == "vec(4'b0001)"
    assert str(uint2vec(2, size=4)) == "vec(4'b0010)"
    assert str(uint2vec(3, size=4)) == "vec(4'b0011)"
    assert str(uint2vec(4, size=4)) == "vec(4'b0100)"
    assert str(uint2vec(5, size=4)) == "vec(4'b0101)"
    assert str(uint2vec(6, size=4)) == "vec(4'b0110)"
    assert str(uint2vec(7, size=4)) == "vec(4'b0111)"
    assert str(uint2vec(8, size=4)) == "vec(4'b1000)"

    with pytest.raises(ValueError):
        uint2vec(8, size=3)


def test_empty():
    """Test empty vector"""
    v = vec()

    # Test properties
    assert v.shape == (0,)
    assert v.data == 0
    assert v.ndim == 1
    assert v.size == 0
    assert v.nbits == 0
    assert list(v.flat) == []  # pylint: disable = use-implicit-booleaness-not-comparison

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

    # Test to_uint
    assert v.to_uint() == 0


def test_scalar():
    """Test scalar (vector w/ one element)"""
    vn = vec(logic.N)
    v0 = vec(0)
    v1 = vec(1)
    vx = vec(logic.X)

    # Test properties
    assert v0.shape == (1,)
    assert v0.data == logic.F.value
    assert v0.ndim == 1
    assert v0.size == 1
    assert v0.nbits == 2
    assert list(v0.flat) == [logic.F]

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

    # Test to_uint
    with pytest.raises(ValueError):
        assert vn.to_uint()
    assert v0.to_uint() == 0
    assert v1.to_uint() == 1
    with pytest.raises(ValueError):
        assert vx.to_uint()


def test_rank1_str():
    """Test vec rank1 string input"""
    v = vec("8'bx10X_x10X")
    data = 0b11100100_11100100
    xs = [
        logic.N,
        logic.F,
        logic.T,
        logic.X,
    ] * 2

    # Test properties
    assert v.shape == (8,)
    assert v.data == data
    assert v.ndim == 1
    assert v.size == 8
    assert v.nbits == 16
    assert list(v.flat) == xs

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec(8'bx10X_x10X)"

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
    assert v == vec("8'bx10X_x10X")
    # Same data, different shape
    assert v != v.reshape((2, 4))
    with pytest.raises(ValueError):
        v.reshape((42,))
    # Different data, same shape
    assert v != vec("8'b0000_0000")

    # Test to_uint
    assert vec("16'b1101_1110_1010_1101").to_uint() == 0xDEAD


def test_rank1_logic():
    xs = [logic.N, logic.F, logic.T, logic.X]
    v1 = vec(xs)
    v2 = vec([0, 1, 0, 1])
    with pytest.raises(TypeError):
        _ = vec([0, "invalid"])

    # Test properties
    assert v1.shape == (4,)
    assert v1.data == 0b11100100
    assert v1.ndim == 1
    assert v1.size == 4
    assert v1.nbits == 8
    assert list(v1.flat) == xs

    # Test __str__ and __repr__
    assert str(v1) == repr(v1) == "vec(4'bx10X)"
    assert str(v2) == repr(v2) == "vec(4'b1010)"


def test_rank2_str():
    v = vec(["4'bx10X", "4'bx10X"])

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([4'bx10X, 4'bx10X])"


def test_rank2_vec():
    v = vec([vec("4'bx10X"), vec("4'bx10X")])

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([4'bx10X, 4'bx10X])"


def test_rank2_errors():
    # Mismatched str literal
    with pytest.raises(TypeError):
        vec(["4'bx10X", "3'b10X"])
    # logicvec followed by some invalid type
    with pytest.raises(TypeError):
        vec(["4'bx10X", 42])


R3VEC = """\
vec([[4'bx10X, 4'bx10X],
     [4'bx10X, 4'bx10X]])"""


def test_rank3_vec():
    v = vec(
        [
            [vec("4'bx10X"), vec("4'bx10X")],
            [vec("4'bx10X"), vec("4'bx10X")],
        ]
    )

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R3VEC


R4VEC = """\
vec([[[4'bx10X, 4'bx10X],
      [4'bx10X, 4'bx10X]],

     [[4'bx10X, 4'bx10X],
      [4'bx10X, 4'bx10X]]])"""


def test_rank4_vec():
    v = vec(
        [
            [[vec("4'bx10X"), vec("4'bx10X")], [vec("4'bx10X"), vec("4'bx10X")]],
            [[vec("4'bx10X"), vec("4'bx10X")], [vec("4'bx10X"), vec("4'bx10X")]],
        ]
    )

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R4VEC


def test_invalid_vec():
    with pytest.raises(TypeError):
        vec(42)


def test_cat():
    assert cat([]) == vec()
    assert cat([False, True, False, True]) == vec("4'b1010")

    with pytest.raises(TypeError):
        cat(["invalid"])

    v = cat([vec("2'b00"), vec("2'b01"), vec("2'b10"), vec("2'b11")], flatten=True)
    assert v == vec("8'b11100100")
    assert v.shape == (8,)

    v = cat([vec("2'b00"), vec("2'b01"), vec("2'b10"), vec("2'b11")], flatten=False)
    assert v.shape == (4, 2)

    v = cat([vec("1'b0"), vec("2'b01"), vec("1'b1"), vec("2'b11")], flatten=True)
    assert v.shape == (6,)

    v = cat([vec("1'b0"), vec("2'b01"), vec("1'b1"), vec("2'b11")], flatten=False)
    assert v.shape == (6,)

    # Incompatible shapes
    with pytest.raises(ValueError):
        cat([vec("2'b00"), vec([vec("2'b00"), vec("2'b00")])])


def test_rep():
    v = rep(vec("2'b00"), 4, flatten=True)
    assert v == vec("8'b0000_0000")
    assert v.shape == (8,)

    v = rep(vec("2'b00"), 4, flatten=False)
    assert v.shape == (4, 2)


def test_consts():
    assert nulls((8,)) == vec("8'bXXXX_XXXX")
    assert zeros((8,)) == vec("8'b0000_0000")
    assert ones((8,)) == vec("8'b1111_1111")
    assert xes((8,)) == vec("8'bxxxx_xxxx")


def test_slicing():
    v = vec(
        [
            [vec("4'b0000"), vec("4'b0001"), vec("4'b0010"), vec("4'b0011")],
            [vec("4'b0100"), vec("4'b0101"), vec("4'b0110"), vec("4'b0111")],
            [vec("4'b1000"), vec("4'b1001"), vec("4'b1010"), vec("4'b1011")],
            [vec("4'b1100"), vec("4'b1101"), vec("4'b1110"), vec("4'b1111")],
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

    assert v[0] == vec([vec("4'b0000"), vec("4'b0001"), vec("4'b0010"), vec("4'b0011")])
    assert v[1] == vec([vec("4'b0100"), vec("4'b0101"), vec("4'b0110"), vec("4'b0111")])
    assert v[2] == vec([vec("4'b1000"), vec("4'b1001"), vec("4'b1010"), vec("4'b1011")])
    assert v[3] == vec([vec("4'b1100"), vec("4'b1101"), vec("4'b1110"), vec("4'b1111")])

    assert v[0, 0] == v[0, 0, :]
    assert v[0, 0] == v[0, 0, 0:4]
    assert v[0, 0] == v[0, 0, -4:]
    assert v[0, 0] == v[0, 0, -5:]

    assert v[0, 0] == vec("4'b0000")
    assert v[1, 1] == vec("4'b0101")
    assert v[2, 2] == vec("4'b1010")
    assert v[3, 3] == vec("4'b1111")

    assert v[0, :, 0] == vec("4'b1010")
    assert v[1, :, 1] == vec("4'b1100")
    assert v[2, :, 2] == vec("4'b0000")
    assert v[3, :, 3] == vec("4'b1111")

    assert v[0, 0, :-1] == vec("3'b000")
    assert v[0, 0, :-2] == vec("2'b00")
    assert v[0, 0, :-3] == vec("1'b0")
    assert v[0, 0, :-4] == vec()

    assert v[0, 0, 0] == logic.F
    assert v[0, vec("2'b00"), 0] == logic.F
    assert v[-4, -4, -4] == logic.F
    assert v[3, 3, 3] == logic.T
    assert v[3, vec("2'b11"), 3] == logic.T
    assert v[-1, -1, -1] == logic.T

    with pytest.raises(ValueError):
        v[0, 0, 0, 0]
    with pytest.raises(TypeError):
        v["invalid"]
