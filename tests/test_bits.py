"""Test bit array data type."""

# pylint: disable = pointless-statement
# pylint: disable = protected-access

import pytest

from seqlogic.bits import F, T, W, X, cat, foo, illogicals, ones, rep, uint2bits, xes, zeros


def test_not():
    """Test bits NOT method."""
    x = foo([W, F, T, X])
    assert str(~x) == "bits(4bX01?)"


def test_nor():
    """Test bits NOR method."""
    x0 = foo("16bXXXX_1111_0000_????")
    x1 = foo("16bX10?_X10?_X10?_X10?")
    assert str(x0.lnor(x1)) == "bits(16bX0X?_000?_X01?_????)"


def test_or():
    """Test bits OR method."""
    x0 = foo("16bXXXX_1111_0000_????")
    x1 = foo("16bX10?_X10?_X10?_X10?")
    assert str(x0 | x1) == "bits(16bX1X?_111?_X10?_????)"


def test_nand():
    """Test bits NAND method."""
    x0 = foo("16bXXXX_1111_0000_????")
    x1 = foo("16bX10?_X10?_X10?_X10?")
    assert str(x0.lnand(x1)) == "bits(16bXX1?_X01?_111?_????)"


def test_and():
    """Test bits AND method."""
    x0 = foo("16bXXXX_1111_0000_????")
    x1 = foo("16bX10?_X10?_X10?_X10?")
    assert str(x0 & x1) == "bits(16bXX0?_X10?_000?_????)"


def test_xnor():
    """Test bits XNOR method."""
    x0 = foo("16bXXXX_1111_0000_????")
    x1 = foo("16bX10?_X10?_X10?_X10?")
    assert str(x0.lxnor(x1)) == "bits(16bXXX?_X10?_X01?_????)"


def test_xor():
    """Test bits XOR method."""
    x0 = foo("16bXXXX_1111_0000_????")
    x1 = foo("16bX10?_X10?_X10?_X10?")
    assert str(x0 ^ x1) == "bits(16bXXX?_X01?_X10?_????)"


def test_uor():
    """Test bits unary OR method."""
    assert foo("2b??").ulor() == W
    assert foo("2b0?").ulor() == W
    assert foo("2b1?").ulor() == W
    assert foo("2bX?").ulor() == W
    assert foo("2b?0").ulor() == W
    assert foo("2b00").ulor() == F
    assert foo("2b10").ulor() == T
    assert foo("2bX0").ulor() == X
    assert foo("2b?1").ulor() == W
    assert foo("2b01").ulor() == T
    assert foo("2b11").ulor() == T
    assert foo("2bX1").ulor() == T
    assert foo("2b?X").ulor() == W
    assert foo("2b0X").ulor() == X
    assert foo("2b1X").ulor() == T
    assert foo("2bXX").ulor() == X


def test_uand():
    """Test bits unary AND method."""
    assert foo("2b??").uland() == W
    assert foo("2b0?").uland() == W
    assert foo("2b1?").uland() == W
    assert foo("2bX?").uland() == W
    assert foo("2b?0").uland() == W
    assert foo("2b00").uland() == F
    assert foo("2b10").uland() == F
    assert foo("2bX0").uland() == F
    assert foo("2b?1").uland() == W
    assert foo("2b01").uland() == F
    assert foo("2b11").uland() == T
    assert foo("2bX1").uland() == X
    assert foo("2b?X").uland() == W
    assert foo("2b0X").uland() == F
    assert foo("2b1X").uland() == X
    assert foo("2bXX").uland() == X


def test_uxor():
    """Test bits unary XOR method."""
    assert foo("2b??").ulxor() == W
    assert foo("2b0?").ulxor() == W
    assert foo("2b1?").ulxor() == W
    assert foo("2bX?").ulxor() == W
    assert foo("2b?0").ulxor() == W
    assert foo("2b00").ulxor() == F
    assert foo("2b10").ulxor() == T
    assert foo("2bX0").ulxor() == X
    assert foo("2b?1").ulxor() == W
    assert foo("2b01").ulxor() == T
    assert foo("2b11").ulxor() == F
    assert foo("2bX1").ulxor() == X
    assert foo("2b?X").ulxor() == W
    assert foo("2b0X").ulxor() == X
    assert foo("2b1X").ulxor() == X
    assert foo("2bXX").ulxor() == X


def test_zext():
    """Test bits zext method."""
    assert foo("4b1010").zext(4) == foo("8b0000_1010")
    # Zero extension on multi-dimensional array will flatten
    assert foo(["4b0000", "4b1111"]).zext(2) == foo("10b00_1111_0000")


def test_sext():
    """Test bits sext method."""
    assert foo("4b1010").sext(4) == foo("8b1111_1010")
    assert foo("4b0101").sext(4) == foo("8b0000_0101")
    # Sign extension of multi-dimensional array will flatten
    assert foo(["4b0000", "4b1111"]).sext(2) == foo("10b11_1111_0000")


def test_lsh():
    """Test bits lsh method."""
    v = foo("4b1111")
    assert v.lsh(0) == (foo("4b1111"), foo())
    assert v.lsh(1) == (foo("4b1110"), foo("1b1"))
    assert v.lsh(2) == (foo("4b1100"), foo("2b11"))
    assert v << 2 == foo("4b1100")
    assert v.lsh(3) == (foo("4b1000"), foo("3b111"))
    assert v.lsh(4) == (foo("4b0000"), foo("4b1111"))
    with pytest.raises(ValueError):
        v.lsh(-1)
    with pytest.raises(ValueError):
        v.lsh(5)

    assert v.lsh(2, foo("2b00")) == (foo("4b1100"), foo("2b11"))
    with pytest.raises(ValueError):
        assert v.lsh(2, foo("3b000"))  # pyright: ignore[reportAssertAlwaysTrue]

    assert foo(["4b0000", "4b1111"]).lsh(2) == (foo("8b1100_0000"), foo("2b11"))


def test_rsh():
    """Test bits rsh method."""
    v = foo("4b1111")
    assert v.rsh(0) == (foo("4b1111"), foo())
    assert v.rsh(1) == (foo("4b0111"), foo("1b1"))
    assert v.rsh(2) == (foo("4b0011"), foo("2b11"))
    assert v >> 2 == foo("4b0011")
    assert v.rsh(3) == (foo("4b0001"), foo("3b111"))
    assert v.rsh(4) == (foo("4b0000"), foo("4b1111"))
    with pytest.raises(ValueError):
        v.rsh(-1)
    with pytest.raises(ValueError):
        v.rsh(5)

    assert v.rsh(2, foo("2b00")) == (foo("4b0011"), foo("2b11"))
    with pytest.raises(ValueError):
        assert v.rsh(2, foo("3b000"))  # pyright: ignore[reportAssertAlwaysTrue]

    assert foo(["4b0000", "4b1111"]).rsh(2) == (foo("8b0011_1100"), foo("2b00"))


def test_arsh():
    """Test bits arsh method."""
    v = foo("4b1111")
    assert v.arsh(0) == (foo("4b1111"), foo())
    assert v.arsh(1) == (foo("4b1111"), foo("1b1"))
    assert v.arsh(2) == (foo("4b1111"), foo("2b11"))
    assert v.arsh(3) == (foo("4b1111"), foo("3b111"))
    assert v.arsh(4) == (foo("4b1111"), foo("4b1111"))

    v = foo("4b0111")
    assert v.arsh(0) == (foo("4b0111"), foo())
    assert v.arsh(1) == (foo("4b0011"), foo("1b1"))
    assert v.arsh(2) == (foo("4b0001"), foo("2b11"))
    assert v.arsh(3) == (foo("4b0000"), foo("3b111"))
    assert v.arsh(4) == (foo("4b0000"), foo("4b0111"))

    with pytest.raises(ValueError):
        v.arsh(-1)
    with pytest.raises(ValueError):
        v.arsh(5)

    assert foo(["4b0000", "4b1111"]).arsh(2) == (foo("8b1111_1100"), foo("2b00"))


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
    """Test bits add method."""
    for a, b, ci, s, co, v in ADD_VALS:
        a, b, s = foo(a), foo(b), foo(s)
        assert a.add(b, ci) == (s, co, v)


def test_addsubops():
    """Test bits add/substract operators."""
    assert foo("2b00") + foo("2b00") == foo("2b00")
    assert foo("2b00") + foo("2b01") == foo("2b01")
    assert foo("2b01") + foo("2b00") == foo("2b01")
    assert foo("2b00") + foo("2b10") == foo("2b10")
    assert foo("2b01") + foo("2b01") == foo("2b10")
    assert foo("2b10") + foo("2b00") == foo("2b10")
    assert foo("2b00") + foo("2b11") == foo("2b11")
    assert foo("2b01") + foo("2b10") == foo("2b11")
    assert foo("2b10") + foo("2b01") == foo("2b11")
    assert foo("2b11") + foo("2b00") == foo("2b11")
    assert foo("2b01") + foo("2b11") == foo("2b00")
    assert foo("2b10") + foo("2b10") == foo("2b00")
    assert foo("2b11") + foo("2b01") == foo("2b00")
    assert foo("2b10") + foo("2b11") == foo("2b01")
    assert foo("2b11") + foo("2b10") == foo("2b01")
    assert foo("2b11") + foo("2b11") == foo("2b10")

    assert foo("2b00") - foo("2b11") == foo("2b01")
    assert foo("2b00") - foo("2b10") == foo("2b10")
    assert foo("2b01") - foo("2b11") == foo("2b10")
    assert foo("2b00") - foo("2b01") == foo("2b11")
    assert foo("2b01") - foo("2b10") == foo("2b11")
    assert foo("2b10") - foo("2b11") == foo("2b11")
    assert foo("2b00") - foo("2b00") == foo("2b00")
    assert foo("2b01") - foo("2b01") == foo("2b00")
    assert foo("2b10") - foo("2b10") == foo("2b00")
    assert foo("2b11") - foo("2b11") == foo("2b00")
    assert foo("2b01") - foo("2b00") == foo("2b01")
    assert foo("2b10") - foo("2b01") == foo("2b01")
    assert foo("2b11") - foo("2b10") == foo("2b01")
    assert foo("2b10") - foo("2b00") == foo("2b10")
    assert foo("2b11") - foo("2b01") == foo("2b10")
    assert foo("2b11") - foo("2b00") == foo("2b11")

    assert -foo("3b000") == foo("3b000")
    assert -foo("3b001") == foo("3b111")
    assert -foo("3b111") == foo("3b001")
    assert -foo("3b010") == foo("3b110")
    assert -foo("3b110") == foo("3b010")
    assert -foo("3b011") == foo("3b101")
    assert -foo("3b101") == foo("3b011")
    assert -foo("3b100") == foo("3b100")


def test_operand_shape_mismatch():
    """Test bits operations with mismatching shapes.

    We could implement something like Verilog's loose typing, but for the time
    being just treat this as illegal.
    """
    x0 = foo("4b1010")
    x1 = foo("8b0101_0101")
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
        foo("4b1010_1010")
    with pytest.raises(ValueError):
        foo("8b1010")
    with pytest.raises(ValueError):
        foo("16hdead_beef")
    with pytest.raises(ValueError):
        foo("8hdead")

    # Invalid input
    with pytest.raises(ValueError):
        foo("invalid")

    # Valid input
    v = foo("4bX1_0?")
    assert v._w.data == 0b11_10_01_00
    v = foo("64hFeDc_Ba98_7654_3210")
    assert v._w.data == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655
    v = foo("64hfEdC_bA98_7654_3210")
    assert v._w.data == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655


def test_uint2vec():
    """Test parsing int literals."""
    with pytest.raises(ValueError):
        uint2bits(-1)

    assert str(uint2bits(0)) == "bits([])"
    assert str(uint2bits(1)) == "bits([1])"
    assert str(uint2bits(2)) == "bits(2b10)"
    assert str(uint2bits(3)) == "bits(2b11)"
    assert str(uint2bits(4)) == "bits(3b100)"
    assert str(uint2bits(5)) == "bits(3b101)"
    assert str(uint2bits(6)) == "bits(3b110)"
    assert str(uint2bits(7)) == "bits(3b111)"
    assert str(uint2bits(8)) == "bits(4b1000)"

    assert str(uint2bits(0, n=4)) == "bits(4b0000)"
    assert str(uint2bits(1, n=4)) == "bits(4b0001)"
    assert str(uint2bits(2, n=4)) == "bits(4b0010)"
    assert str(uint2bits(3, n=4)) == "bits(4b0011)"
    assert str(uint2bits(4, n=4)) == "bits(4b0100)"
    assert str(uint2bits(5, n=4)) == "bits(4b0101)"
    assert str(uint2bits(6, n=4)) == "bits(4b0110)"
    assert str(uint2bits(7, n=4)) == "bits(4b0111)"
    assert str(uint2bits(8, n=4)) == "bits(4b1000)"

    with pytest.raises(ValueError):
        uint2bits(8, n=3)


def test_empty():
    """Test empty vector."""
    v = foo()

    # Test properties
    assert v.shape == (0,)
    assert v._w.data == 0
    assert v.ndim == 1
    assert v.size == 0
    assert v._w.nbits == 0
    assert list(v.flat) == []  # pylint: disable = use-implicit-booleaness-not-comparison

    assert v.flatten() == v

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "bits([])"

    # Test __len__
    assert len(v) == 0

    # Test __iter__
    assert list(v) == []  # pylint: disable = use-implicit-booleaness-not-comparison

    # Test __getitem__
    with pytest.raises(IndexError):
        _ = v[0]

    # Test __eq__
    assert v == foo()
    assert v == v.reshape((0,))
    assert v != foo(0)

    # Test to_uint, to_int
    assert v.to_uint() == 0
    assert v.to_int() == 0


def test_scalar():
    """Test scalar (vector w/ one element)."""
    vn = W
    v0 = foo(0)
    v1 = foo(1)
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
    assert str(vn) == repr(vn) == "bits([?])"
    assert str(v0) == repr(v0) == "bits([0])"
    assert str(v1) == repr(v1) == "bits([1])"
    assert str(vx) == repr(vx) == "bits([X])"

    # Test __len__
    assert len(v0) == 1

    # Test __iter__
    assert list(v0) == [F]

    # Test __getitem__
    assert v0[0] == F

    # Test __eq__
    assert v0 == foo(0)
    assert v0 == v0.reshape((1,))
    assert v0 != foo()
    assert v0 != foo(1)

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
    """Test bits rank1 string input."""
    v = foo("8bX10?_X10?")
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
    assert str(v) == repr(v) == "bits(8bX10?_X10?)"

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
    assert v == foo("8bX10?_X10?")
    # Same data, different shape
    assert v != v.reshape((2, 4))
    with pytest.raises(ValueError):
        v.reshape((42,))
    # Different data, same shape
    assert v != foo("8b0000_0000")

    # Test to_uint
    assert foo("16b1101_1110_1010_1101").to_uint() == 0xDEAD

    # Test to_int
    assert foo("4b0000").to_int() == 0
    assert foo("4b0001").to_int() == 1
    assert foo("4b0110").to_int() == 6
    assert foo("4b0111").to_int() == 7
    assert foo("4b1000").to_int() == -8
    assert foo("4b1001").to_int() == -7
    assert foo("4b1110").to_int() == -2
    assert foo("4b1111").to_int() == -1


def test_rank1_logic():
    """Test foo function w/ rank1 logic input."""
    xs = [W, F, T, X]
    v1 = foo("4bX10?")
    v2 = foo([0, 1, 0, 1])

    # Test properties
    assert v1.shape == (4,)
    assert v1._w.data == 0b11100100
    assert v1.ndim == 1
    assert v1.size == 4
    assert v1._w.nbits == 8
    assert list(v1.flat) == xs

    # Test __str__ and __repr__
    assert str(v1) == repr(v1) == "bits(4bX10?)"
    assert str(v2) == repr(v2) == "bits(4b1010)"


def test_rank2_str():
    """Test foo function w/ rank2 str input."""
    v = foo(["4bX10?", "4bX10?"])

    assert v.flatten() == foo("8bX10?_X10?")

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "bits([4bX10?, 4bX10?])"


def test_rank2_vec():
    """Test foo function w/ rank2 bits input."""
    v = foo([foo("4bX10?"), foo("4bX10?")])

    # Test __str__ and __repr__
    assert str(v) == repr(v) == "bits([4bX10?, 4bX10?])"


def test_rank2_errors():
    """Test foo function rank2 errors."""
    # Mismatched str literal
    with pytest.raises(TypeError):
        foo(["4bX10?", "3b10?"])
    # bits followed by some invalid type
    with pytest.raises(TypeError):
        foo(["4bX10?", 42])


R3VEC = """\
bits([[4bX10?, 4bX10?],
      [4bX10?, 4bX10?]])"""


def test_rank3_vec():
    """Test foo function w/ rank3 input."""
    v = foo(
        [
            [foo("4bX10?"), foo("4bX10?")],
            [foo("4bX10?"), foo("4bX10?")],
        ]
    )

    assert v.flatten() == foo("16bX10?_X10?_X10?_X10?")

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R3VEC


R4VEC = """\
bits([[[4bX10?, 4bX10?],
       [4bX10?, 4bX10?]],

      [[4bX10?, 4bX10?],
       [4bX10?, 4bX10?]]])"""


def test_rank4_vec():
    """Test foo function w/ rank4 input."""
    v = foo(
        [
            [[foo("4bX10?"), foo("4bX10?")], [foo("4bX10?"), foo("4bX10?")]],
            [[foo("4bX10?"), foo("4bX10?")], [foo("4bX10?"), foo("4bX10?")]],
        ]
    )

    # Test __str__ and __repr__
    assert str(v) == repr(v) == R4VEC


def test_invalid_vec():
    """Test foo function invalid input."""
    with pytest.raises(TypeError):
        foo(42)


def test_cat():
    """Test bits cat function."""
    assert cat([]) == foo()
    assert cat([False, True, False, True]) == foo("4b1010")

    with pytest.raises(TypeError):
        cat(["invalid"])  # pyright: ignore[reportArgumentType]

    v = cat([foo("2b00"), foo("2b01"), foo("2b10"), foo("2b11")], flatten=True)
    assert v == foo("8b11100100")
    assert v.shape == (8,)

    v = cat([foo("2b00"), foo("2b01"), foo("2b10"), foo("2b11")], flatten=False)
    assert v.shape == (4, 2)

    v = cat([foo("1b0"), foo("2b01"), foo("1b1"), foo("2b11")], flatten=True)
    assert v.shape == (6,)

    v = cat([foo("1b0"), foo("2b01"), foo("1b1"), foo("2b11")], flatten=False)
    assert v.shape == (6,)

    # Incompatible shapes
    with pytest.raises(ValueError):
        cat([foo("2b00"), foo([foo("2b00"), foo("2b00")])])


def test_rep():
    """Test foo rep function."""
    v = rep(foo("2b00"), 4, flatten=True)
    assert v == foo("8b0000_0000")
    assert v.shape == (8,)

    v = rep(foo("2b00"), 4, flatten=False)
    assert v.shape == (4, 2)


def test_consts():
    """Test bits constants."""
    assert illogicals((8,)) == foo("8b????_????")
    assert zeros((8,)) == foo("8b0000_0000")
    assert ones((8,)) == foo("8b1111_1111")
    assert xes((8,)) == foo("8bXXXX_XXXX")


def test_slicing():
    """Test bits slicing behavior."""
    v = foo(
        [
            [foo("4b0000"), foo("4b0001"), foo("4b0010"), foo("4b0011")],
            [foo("4b0100"), foo("4b0101"), foo("4b0110"), foo("4b0111")],
            [foo("4b1000"), foo("4b1001"), foo("4b1010"), foo("4b1011")],
            [foo("4b1100"), foo("4b1101"), foo("4b1110"), foo("4b1111")],
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

    assert v[0] == foo([foo("4b0000"), foo("4b0001"), foo("4b0010"), foo("4b0011")])
    assert v[1] == foo([foo("4b0100"), foo("4b0101"), foo("4b0110"), foo("4b0111")])
    assert v[2] == foo([foo("4b1000"), foo("4b1001"), foo("4b1010"), foo("4b1011")])
    assert v[3] == foo([foo("4b1100"), foo("4b1101"), foo("4b1110"), foo("4b1111")])

    assert v[0, 0] == v[0, 0, :]
    assert v[0, 0] == v[0, 0, 0:4]
    assert v[0, 0] == v[0, 0, -4:]
    assert v[0, 0] == v[0, 0, -5:]

    assert v[0, 0] == foo("4b0000")
    assert v[1, 1] == foo("4b0101")
    assert v[2, 2] == foo("4b1010")
    assert v[3, 3] == foo("4b1111")

    assert v[0, :, 0] == foo("4b1010")
    assert v[1, :, 1] == foo("4b1100")
    assert v[2, :, 2] == foo("4b0000")
    assert v[3, :, 3] == foo("4b1111")

    assert v[0, 0, :-1] == foo("3b000")
    assert v[0, 0, :-2] == foo("2b00")
    assert v[0, 0, :-3] == foo("1b0")
    assert v[0, 0, :-4] == foo()

    assert v[0, 0, 0] == F
    assert v[0, foo("2b00"), 0] == F
    assert v[-4, -4, -4] == F
    assert v[3, 3, 3] == T
    assert v[3, foo("2b11"), 3] == T
    assert v[-1, -1, -1] == T

    with pytest.raises(ValueError):
        v[0, 0, 0, 0]
    with pytest.raises(TypeError):
        v["invalid"]  # pyright: ignore[reportArgumentType]


def test_countbits():
    """Test bits countbits methods."""
    v = foo("8bX10?_X10?")
    assert v._w.count_illogicals() == 2
    assert v._w.count_zeros() == 2
    assert v._w.count_ones() == 2
    assert v._w.count_unknowns() == 2

    assert foo("4b0000")._w.count_ones() == 0
    assert foo("4b0001")._w.count_ones() == 1
    assert foo("4b0011")._w.count_ones() == 2
    assert foo("4b0111")._w.count_ones() == 3
    assert foo("4b1111")._w.count_ones() == 4

    assert not foo("4b0000")._w.onehot()
    assert foo("4b1000")._w.onehot()
    assert foo("4b0001")._w.onehot()
    assert not foo("4b1001")._w.onehot()
    assert not foo("4b1101")._w.onehot()

    assert foo("4b0000")._w.onehot0()
    assert foo("4b1000")._w.onehot0()
    assert not foo("4b1010")._w.onehot0()
    assert not foo("4b1011")._w.onehot0()

    assert not foo("4b0000")._w.has_unknown()
    assert not foo("4b1111")._w.has_unknown()
    assert foo("4b0X01")._w.has_unknown()
    assert foo("4b01?1")._w.has_illogical()
