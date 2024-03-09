"""Test Logic Vector data type."""

# pylint: disable = pointless-statement
# pylint: disable = protected-access

import pytest

from seqlogic.bits import F, T, W, X, cat, illogicals, ones, rep, uint2vec, vec, xes, zeros


def test_not():
    """Test logicvec NOT function."""
    x = vec([W, F, T, X])
    assert str(~x) == "vec(4bX01?)"


def test_nor():
    """Test logicvec NOR function."""
    x0 = vec("16bXXXX_1111_0000_????")
    x1 = vec("16bX10?_X10?_X10?_X10?")
    assert str(x0.lnor(x1)) == "vec(16bX0X?_000?_X01?_????)"


def test_or():
    """Test logicvec OR function."""
    x0 = vec("16bXXXX_1111_0000_????")
    x1 = vec("16bX10?_X10?_X10?_X10?")
    assert str(x0 | x1) == "vec(16bX1X?_111?_X10?_????)"


def test_nand():
    """Test logicvec NAND function."""
    x0 = vec("16bXXXX_1111_0000_????")
    x1 = vec("16bX10?_X10?_X10?_X10?")
    assert str(x0.lnand(x1)) == "vec(16bXX1?_X01?_111?_????)"


def test_and():
    """Test logicvec AND function."""
    x0 = vec("16bXXXX_1111_0000_????")
    x1 = vec("16bX10?_X10?_X10?_X10?")
    assert str(x0 & x1) == "vec(16bXX0?_X10?_000?_????)"


def test_xnor():
    """Test logicvec XNOR function."""
    x0 = vec("16bXXXX_1111_0000_????")
    x1 = vec("16bX10?_X10?_X10?_X10?")
    assert str(x0.lxnor(x1)) == "vec(16bXXX?_X10?_X01?_????)"


def test_xor():
    """Test logicvec XOR function."""
    x0 = vec("16bXXXX_1111_0000_????")
    x1 = vec("16bX10?_X10?_X10?_X10?")
    assert str(x0 ^ x1) == "vec(16bXXX?_X01?_X10?_????)"


def test_uor():
    """Test logicvec unary OR method."""
    assert vec("2b??").ulor() == W
    assert vec("2b0?").ulor() == W
    assert vec("2b1?").ulor() == W
    assert vec("2bX?").ulor() == W
    assert vec("2b?0").ulor() == W
    assert vec("2b00").ulor() == F
    assert vec("2b10").ulor() == T
    assert vec("2bX0").ulor() == X
    assert vec("2b?1").ulor() == W
    assert vec("2b01").ulor() == T
    assert vec("2b11").ulor() == T
    assert vec("2bX1").ulor() == T
    assert vec("2b?X").ulor() == W
    assert vec("2b0X").ulor() == X
    assert vec("2b1X").ulor() == T
    assert vec("2bXX").ulor() == X


def test_uand():
    """Test logicvec unary and method."""
    assert vec("2b??").uland() == W
    assert vec("2b0?").uland() == W
    assert vec("2b1?").uland() == W
    assert vec("2bX?").uland() == W
    assert vec("2b?0").uland() == W
    assert vec("2b00").uland() == F
    assert vec("2b10").uland() == F
    assert vec("2bX0").uland() == F
    assert vec("2b?1").uland() == W
    assert vec("2b01").uland() == F
    assert vec("2b11").uland() == T
    assert vec("2bX1").uland() == X
    assert vec("2b?X").uland() == W
    assert vec("2b0X").uland() == F
    assert vec("2b1X").uland() == X
    assert vec("2bXX").uland() == X


def test_uxor():
    """Test logicvec unary xor method."""
    assert vec("2b??").ulxor() == W
    assert vec("2b0?").ulxor() == W
    assert vec("2b1?").ulxor() == W
    assert vec("2bX?").ulxor() == W
    assert vec("2b?0").ulxor() == W
    assert vec("2b00").ulxor() == F
    assert vec("2b10").ulxor() == T
    assert vec("2bX0").ulxor() == X
    assert vec("2b?1").ulxor() == W
    assert vec("2b01").ulxor() == T
    assert vec("2b11").ulxor() == F
    assert vec("2bX1").ulxor() == X
    assert vec("2b?X").ulxor() == W
    assert vec("2b0X").ulxor() == X
    assert vec("2b1X").ulxor() == X
    assert vec("2bXX").ulxor() == X


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
        assert v.lsh(2, vec("3b000"))  # pyright: ignore[reportAssertAlwaysTrue]

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
        assert v.rsh(2, vec("3b000"))  # pyright: ignore[reportAssertAlwaysTrue]

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
    ("2b00", "2b00", F, "2b00", F, F),
    ("2b00", "2b01", 0, "2b01", F, F),
    ("2b00", "2b10", F, "2b10", F, F),
    ("2b00", "2b11", 0, "2b11", F, F),
    ("2b01", "2b00", F, "2b01", F, F),
    ("2b01", "2b01", 0, "2b10", F, T),  # overflow
    ("2b01", "2b10", F, "2b11", F, F),
    ("2b01", "2b11", 0, "2b00", T, F),
    ("2b10", "2b00", F, "2b10", F, F),
    ("2b10", "2b01", 0, "2b11", F, F),
    ("2b10", "2b10", F, "2b00", T, T),  # overflow
    ("2b10", "2b11", 0, "2b01", T, T),  # overflow
    ("2b11", "2b00", F, "2b11", F, F),
    ("2b11", "2b01", 0, "2b00", T, F),
    ("2b11", "2b10", F, "2b01", T, T),  # overflow
    ("2b11", "2b11", 0, "2b10", T, F),
    ("2b00", "2b00", T, "2b01", F, F),
    ("2b00", "2b01", 1, "2b10", F, T),  # overflow
    ("2b00", "2b10", T, "2b11", F, F),
    ("2b00", "2b11", 1, "2b00", T, F),
    ("2b01", "2b00", T, "2b10", F, T),  # overflow
    ("2b01", "2b01", 1, "2b11", F, T),  # overflow
    ("2b01", "2b10", T, "2b00", T, F),
    ("2b01", "2b11", 1, "2b01", T, F),
    ("2b10", "2b00", T, "2b11", F, F),
    ("2b10", "2b01", 1, "2b00", T, F),
    ("2b10", "2b10", T, "2b01", T, T),  # overflow
    ("2b10", "2b11", 1, "2b10", T, F),
    ("2b11", "2b00", T, "2b00", T, F),
    ("2b11", "2b01", 1, "2b01", T, F),
    ("2b11", "2b10", T, "2b10", T, F),
    ("2b11", "2b11", 1, "2b11", T, F),
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
        x0 | x1  # pyright: ignore[reportUnusedExpression]
    with pytest.raises(ValueError):
        x0.lnand(x1)
    with pytest.raises(ValueError):
        x0 & x1  # pyright: ignore[reportUnusedExpression]
    with pytest.raises(ValueError):
        x0.lxnor(x1)
    with pytest.raises(ValueError):
        x0 ^ x1  # pyright: ignore[reportUnusedExpression]


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
    v = vec("4bX1_0?")
    assert v._w.data == 0b11_10_01_00
    v = vec("64hFeDc_Ba98_7654_3210")
    assert v._w.data == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655
    v = vec("64hfEdC_bA98_7654_3210")
    assert v._w.data == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655


def test_uint2vec():
    """Test parsing int literals."""
    with pytest.raises(ValueError):
        uint2vec(-1)

    assert str(uint2vec(0)) == "vec([])"
    assert str(uint2vec(1)) == "vec([1])"
    assert str(uint2vec(2)) == "vec(2b10)"
    assert str(uint2vec(3)) == "vec(2b11)"
    assert str(uint2vec(4)) == "vec(3b100)"
    assert str(uint2vec(5)) == "vec(3b101)"
    assert str(uint2vec(6)) == "vec(3b110)"
    assert str(uint2vec(7)) == "vec(3b111)"
    assert str(uint2vec(8)) == "vec(4b1000)"

    assert str(uint2vec(0, n=4)) == "vec(4b0000)"
    assert str(uint2vec(1, n=4)) == "vec(4b0001)"
    assert str(uint2vec(2, n=4)) == "vec(4b0010)"
    assert str(uint2vec(3, n=4)) == "vec(4b0011)"
    assert str(uint2vec(4, n=4)) == "vec(4b0100)"
    assert str(uint2vec(5, n=4)) == "vec(4b0101)"
    assert str(uint2vec(6, n=4)) == "vec(4b0110)"
    assert str(uint2vec(7, n=4)) == "vec(4b0111)"
    assert str(uint2vec(8, n=4)) == "vec(4b1000)"

    with pytest.raises(ValueError):
        uint2vec(8, n=3)


def test_empty():
    """Test empty vector."""
    v = vec()

    # Test properties
    assert v.shape == (0,)
    assert v._w.data == 0
    assert v.ndim == 1
    assert v.size == 0
    assert v._w.nbits == 0
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
    vn = W
    v0 = vec(0)
    v1 = vec(1)
    vx = X

    # Test properties
    assert v0.shape == (1,)
    assert v0._w.data == 0b01
    assert v0.ndim == 1
    assert v0.size == 1
    assert v0._w.nbits == 2
    assert list(v0.flat) == [F]

    assert v0.flatten() == v0

    # Test __str__ and __repr__
    assert str(vn) == repr(vn) == "vec([?])"
    assert str(v0) == repr(v0) == "vec([0])"
    assert str(v1) == repr(v1) == "vec([1])"
    assert str(vx) == repr(vx) == "vec([X])"

    # Test __len__
    assert len(v0) == 1

    # Test __iter__
    assert list(v0) == [F]

    # Test __getitem__
    assert v0[0] == F

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
    v = vec("8bX10?_X10?")
    data = 0b11100100_11100100
    xs = [W, F, T, X] * 2

    # Test properties
    assert v.shape == (8,)
    assert v._w.data == data
    assert v.ndim == 1
    assert v.size == 8
    assert v._w.nbits == 16
    assert list(v.flat) == xs

    assert v.flatten() == v

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec(8bX10?_X10?)"

    # Test __len__
    assert len(v) == 8

    # Test __iter__
    assert list(v) == xs

    # Test __getitem__
    assert v[0] == W
    assert v[1] == F
    assert v[6] == T
    assert v[7] == X

    # Test __eq__
    assert v == vec("8bX10?_X10?")
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
    xs = [W, F, T, X]
    v1 = vec("4bX10?")
    v2 = vec([0, 1, 0, 1])

    # Test properties
    assert v1.shape == (4,)
    assert v1._w.data == 0b11100100
    assert v1.ndim == 1
    assert v1.size == 4
    assert v1._w.nbits == 8
    assert list(v1.flat) == xs

    # Test __str__ and __repr__
    assert str(v1) == repr(v1) == "vec(4bX10?)"
    assert str(v2) == repr(v2) == "vec(4b1010)"


def test_rank2_str():
    """Test vec function w/ rank2 str input."""
    v = vec(["4bX10?", "4bX10?"])

    assert v.flatten() == vec("8bX10?_X10?")

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([4bX10?, 4bX10?])"


def test_rank2_vec():
    """Test vec function w/ rank2 vec input."""
    v = vec([vec("4bX10?"), vec("4bX10?")])

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "vec([4bX10?, 4bX10?])"


def test_rank2_errors():
    """Test vec function rank2 errors."""
    # Mismatched str literal
    with pytest.raises(TypeError):
        vec(["4bX10?", "3b10?"])
    # logicvec followed by some invalid type
    with pytest.raises(TypeError):
        vec(["4bX10?", 42])


R3VEC = """\
vec([[4bX10?, 4bX10?],
     [4bX10?, 4bX10?]])"""


def test_rank3_vec():
    """Test vec function w/ rank3 input."""
    v = vec(
        [
            [vec("4bX10?"), vec("4bX10?")],
            [vec("4bX10?"), vec("4bX10?")],
        ]
    )

    assert v.flatten() == vec("16bX10?_X10?_X10?_X10?")

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R3VEC


R4VEC = """\
vec([[[4bX10?, 4bX10?],
      [4bX10?, 4bX10?]],

     [[4bX10?, 4bX10?],
      [4bX10?, 4bX10?]]])"""


def test_rank4_vec():
    """Test vec function w/ rank4 input."""
    v = vec(
        [
            [[vec("4bX10?"), vec("4bX10?")], [vec("4bX10?"), vec("4bX10?")]],
            [[vec("4bX10?"), vec("4bX10?")], [vec("4bX10?"), vec("4bX10?")]],
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
        cat(["invalid"])  # pyright: ignore[reportArgumentType]

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
    assert illogicals((8,)) == vec("8b????_????")
    assert zeros((8,)) == vec("8b0000_0000")
    assert ones((8,)) == vec("8b1111_1111")
    assert xes((8,)) == vec("8bXXXX_XXXX")


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
        v["invalid"]  # pyright: ignore[reportArgumentType]

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

    assert v[0, 0, 0] == F
    assert v[0, vec("2b00"), 0] == F
    assert v[-4, -4, -4] == F
    assert v[3, 3, 3] == T
    assert v[3, vec("2b11"), 3] == T
    assert v[-1, -1, -1] == T

    with pytest.raises(ValueError):
        v[0, 0, 0, 0]
    with pytest.raises(TypeError):
        v["invalid"]  # pyright: ignore[reportArgumentType]


def test_countbits():
    """Test logicvec countbits methods."""
    v = vec("8bX10?_X10?")
    assert v._w.count_illogicals() == 2
    assert v._w.count_zeros() == 2
    assert v._w.count_ones() == 2
    assert v._w.count_unknowns() == 2

    assert vec("4b0000")._w.count_ones() == 0
    assert vec("4b0001")._w.count_ones() == 1
    assert vec("4b0011")._w.count_ones() == 2
    assert vec("4b0111")._w.count_ones() == 3
    assert vec("4b1111")._w.count_ones() == 4

    assert not vec("4b0000")._w.onehot()
    assert vec("4b1000")._w.onehot()
    assert vec("4b0001")._w.onehot()
    assert not vec("4b1001")._w.onehot()
    assert not vec("4b1101")._w.onehot()

    assert vec("4b0000")._w.onehot0()
    assert vec("4b1000")._w.onehot0()
    assert not vec("4b1010")._w.onehot0()
    assert not vec("4b1011")._w.onehot0()

    assert not vec("4b0000")._w.has_unknown()
    assert not vec("4b1111")._w.has_unknown()
    assert vec("4b0X01")._w.has_unknown()
    assert vec("4b01?1")._w.has_illogical()
