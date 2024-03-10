"""Test bit array data type."""

# pylint: disable = pointless-statement
# pylint: disable = protected-access

import pytest

from seqlogic.bits import F, T, W, X, bits, cat, illogicals, ones, rep, uint2bits, xes, zeros


def test_not():
    """Test bits NOT method."""
    b = bits([W, F, T, X])
    assert str(~b) == "bits(4bX01?)"


def test_nor():
    """Test bits NOR method."""
    b0 = bits("16bXXXX_1111_0000_????")
    b1 = bits("16bX10?_X10?_X10?_X10?")
    assert str(b0.lnor(b1)) == "bits(16bX0X?_000?_X01?_????)"


def test_or():
    """Test bits OR method."""
    b0 = bits("16bXXXX_1111_0000_????")
    b1 = bits("16bX10?_X10?_X10?_X10?")
    assert str(b0 | b1) == "bits(16bX1X?_111?_X10?_????)"


def test_nand():
    """Test bits NAND method."""
    b0 = bits("16bXXXX_1111_0000_????")
    b1 = bits("16bX10?_X10?_X10?_X10?")
    assert str(b0.lnand(b1)) == "bits(16bXX1?_X01?_111?_????)"


def test_and():
    """Test bits AND method."""
    b0 = bits("16bXXXX_1111_0000_????")
    b1 = bits("16bX10?_X10?_X10?_X10?")
    assert str(b0 & b1) == "bits(16bXX0?_X10?_000?_????)"


def test_xnor():
    """Test bits XNOR method."""
    b0 = bits("16bXXXX_1111_0000_????")
    b1 = bits("16bX10?_X10?_X10?_X10?")
    assert str(b0.lxnor(b1)) == "bits(16bXXX?_X10?_X01?_????)"


def test_xor():
    """Test bits XOR method."""
    b0 = bits("16bXXXX_1111_0000_????")
    b1 = bits("16bX10?_X10?_X10?_X10?")
    assert str(b0 ^ b1) == "bits(16bXXX?_X01?_X10?_????)"


def test_uor():
    """Test bits unary OR method."""
    assert bits("2b??").ulor() == W
    assert bits("2b0?").ulor() == W
    assert bits("2b1?").ulor() == W
    assert bits("2bX?").ulor() == W
    assert bits("2b?0").ulor() == W
    assert bits("2b00").ulor() == F
    assert bits("2b10").ulor() == T
    assert bits("2bX0").ulor() == X
    assert bits("2b?1").ulor() == W
    assert bits("2b01").ulor() == T
    assert bits("2b11").ulor() == T
    assert bits("2bX1").ulor() == T
    assert bits("2b?X").ulor() == W
    assert bits("2b0X").ulor() == X
    assert bits("2b1X").ulor() == T
    assert bits("2bXX").ulor() == X


def test_uand():
    """Test bits unary AND method."""
    assert bits("2b??").uland() == W
    assert bits("2b0?").uland() == W
    assert bits("2b1?").uland() == W
    assert bits("2bX?").uland() == W
    assert bits("2b?0").uland() == W
    assert bits("2b00").uland() == F
    assert bits("2b10").uland() == F
    assert bits("2bX0").uland() == F
    assert bits("2b?1").uland() == W
    assert bits("2b01").uland() == F
    assert bits("2b11").uland() == T
    assert bits("2bX1").uland() == X
    assert bits("2b?X").uland() == W
    assert bits("2b0X").uland() == F
    assert bits("2b1X").uland() == X
    assert bits("2bXX").uland() == X


def test_uxor():
    """Test bits unary XOR method."""
    assert bits("2b??").ulxor() == W
    assert bits("2b0?").ulxor() == W
    assert bits("2b1?").ulxor() == W
    assert bits("2bX?").ulxor() == W
    assert bits("2b?0").ulxor() == W
    assert bits("2b00").ulxor() == F
    assert bits("2b10").ulxor() == T
    assert bits("2bX0").ulxor() == X
    assert bits("2b?1").ulxor() == W
    assert bits("2b01").ulxor() == T
    assert bits("2b11").ulxor() == F
    assert bits("2bX1").ulxor() == X
    assert bits("2b?X").ulxor() == W
    assert bits("2b0X").ulxor() == X
    assert bits("2b1X").ulxor() == X
    assert bits("2bXX").ulxor() == X


def test_zext():
    """Test bits zext method."""
    assert bits("4b1010").zext(4) == bits("8b0000_1010")
    # Zero extension on multi-dimensional array will flatten
    assert bits(["4b0000", "4b1111"]).zext(2) == bits("10b00_1111_0000")


def test_sext():
    """Test bits sext method."""
    assert bits("4b1010").sext(4) == bits("8b1111_1010")
    assert bits("4b0101").sext(4) == bits("8b0000_0101")
    # Sign extension of multi-dimensional array will flatten
    assert bits(["4b0000", "4b1111"]).sext(2) == bits("10b11_1111_0000")


def test_lsh():
    """Test bits lsh method."""
    b = bits("4b1111")
    assert b.lsh(0) == (bits("4b1111"), bits())
    assert b.lsh(1) == (bits("4b1110"), bits("1b1"))
    assert b.lsh(2) == (bits("4b1100"), bits("2b11"))
    assert b << 2 == bits("4b1100")
    assert b.lsh(3) == (bits("4b1000"), bits("3b111"))
    assert b.lsh(4) == (bits("4b0000"), bits("4b1111"))
    with pytest.raises(ValueError):
        b.lsh(-1)
    with pytest.raises(ValueError):
        b.lsh(5)

    assert b.lsh(2, bits("2b00")) == (bits("4b1100"), bits("2b11"))
    with pytest.raises(ValueError):
        assert b.lsh(2, bits("3b000"))  # pyright: ignore[reportAssertAlwaysTrue]

    assert bits(["4b0000", "4b1111"]).lsh(2) == (bits("8b1100_0000"), bits("2b11"))


def test_rsh():
    """Test bits rsh method."""
    b = bits("4b1111")
    assert b.rsh(0) == (bits("4b1111"), bits())
    assert b.rsh(1) == (bits("4b0111"), bits("1b1"))
    assert b.rsh(2) == (bits("4b0011"), bits("2b11"))
    assert b >> 2 == bits("4b0011")
    assert b.rsh(3) == (bits("4b0001"), bits("3b111"))
    assert b.rsh(4) == (bits("4b0000"), bits("4b1111"))
    with pytest.raises(ValueError):
        b.rsh(-1)
    with pytest.raises(ValueError):
        b.rsh(5)

    assert b.rsh(2, bits("2b00")) == (bits("4b0011"), bits("2b11"))
    with pytest.raises(ValueError):
        assert b.rsh(2, bits("3b000"))  # pyright: ignore[reportAssertAlwaysTrue]

    assert bits(["4b0000", "4b1111"]).rsh(2) == (bits("8b0011_1100"), bits("2b00"))


def test_arsh():
    """Test bits arsh method."""
    b = bits("4b1111")
    assert b.arsh(0) == (bits("4b1111"), bits())
    assert b.arsh(1) == (bits("4b1111"), bits("1b1"))
    assert b.arsh(2) == (bits("4b1111"), bits("2b11"))
    assert b.arsh(3) == (bits("4b1111"), bits("3b111"))
    assert b.arsh(4) == (bits("4b1111"), bits("4b1111"))

    b = bits("4b0111")
    assert b.arsh(0) == (bits("4b0111"), bits())
    assert b.arsh(1) == (bits("4b0011"), bits("1b1"))
    assert b.arsh(2) == (bits("4b0001"), bits("2b11"))
    assert b.arsh(3) == (bits("4b0000"), bits("3b111"))
    assert b.arsh(4) == (bits("4b0000"), bits("4b0111"))

    with pytest.raises(ValueError):
        b.arsh(-1)
    with pytest.raises(ValueError):
        b.arsh(5)

    assert bits(["4b0000", "4b1111"]).arsh(2) == (bits("8b1111_1100"), bits("2b00"))


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
        a, b, s = bits(a), bits(b), bits(s)
        assert a.add(b, ci) == (s, co, v)


def test_addsubops():
    """Test bits add/substract operators."""
    assert bits("2b00") + bits("2b00") == bits("2b00")
    assert bits("2b00") + bits("2b01") == bits("2b01")
    assert bits("2b01") + bits("2b00") == bits("2b01")
    assert bits("2b00") + bits("2b10") == bits("2b10")
    assert bits("2b01") + bits("2b01") == bits("2b10")
    assert bits("2b10") + bits("2b00") == bits("2b10")
    assert bits("2b00") + bits("2b11") == bits("2b11")
    assert bits("2b01") + bits("2b10") == bits("2b11")
    assert bits("2b10") + bits("2b01") == bits("2b11")
    assert bits("2b11") + bits("2b00") == bits("2b11")
    assert bits("2b01") + bits("2b11") == bits("2b00")
    assert bits("2b10") + bits("2b10") == bits("2b00")
    assert bits("2b11") + bits("2b01") == bits("2b00")
    assert bits("2b10") + bits("2b11") == bits("2b01")
    assert bits("2b11") + bits("2b10") == bits("2b01")
    assert bits("2b11") + bits("2b11") == bits("2b10")

    assert bits("2b00") - bits("2b11") == bits("2b01")
    assert bits("2b00") - bits("2b10") == bits("2b10")
    assert bits("2b01") - bits("2b11") == bits("2b10")
    assert bits("2b00") - bits("2b01") == bits("2b11")
    assert bits("2b01") - bits("2b10") == bits("2b11")
    assert bits("2b10") - bits("2b11") == bits("2b11")
    assert bits("2b00") - bits("2b00") == bits("2b00")
    assert bits("2b01") - bits("2b01") == bits("2b00")
    assert bits("2b10") - bits("2b10") == bits("2b00")
    assert bits("2b11") - bits("2b11") == bits("2b00")
    assert bits("2b01") - bits("2b00") == bits("2b01")
    assert bits("2b10") - bits("2b01") == bits("2b01")
    assert bits("2b11") - bits("2b10") == bits("2b01")
    assert bits("2b10") - bits("2b00") == bits("2b10")
    assert bits("2b11") - bits("2b01") == bits("2b10")
    assert bits("2b11") - bits("2b00") == bits("2b11")

    assert -bits("3b000") == bits("3b000")
    assert -bits("3b001") == bits("3b111")
    assert -bits("3b111") == bits("3b001")
    assert -bits("3b010") == bits("3b110")
    assert -bits("3b110") == bits("3b010")
    assert -bits("3b011") == bits("3b101")
    assert -bits("3b101") == bits("3b011")
    assert -bits("3b100") == bits("3b100")


def test_operand_shape_mismatch():
    """Test bits operations with mismatching shapes.

    We could implement something like Verilog's loose typing, but for the time
    being just treat this as illegal.
    """
    b0 = bits("4b1010")
    b1 = bits("8b0101_0101")
    with pytest.raises(ValueError):
        b0.lnor(b1)
    with pytest.raises(ValueError):
        b0 | b1  # pyright: ignore[reportUnusedExpression]
    with pytest.raises(ValueError):
        b0.lnand(b1)
    with pytest.raises(ValueError):
        b0 & b1  # pyright: ignore[reportUnusedExpression]
    with pytest.raises(ValueError):
        b0.lxnor(b1)
    with pytest.raises(ValueError):
        b0 ^ b1  # pyright: ignore[reportUnusedExpression]


def test_parse_str_literal():
    """Test parsing of vector string literals."""
    # literal doesn't match size
    with pytest.raises(ValueError):
        bits("4b1010_1010")
    with pytest.raises(ValueError):
        bits("8b1010")
    with pytest.raises(ValueError):
        bits("16hdead_beef")
    with pytest.raises(ValueError):
        bits("8hdead")

    # Invalid input
    with pytest.raises(ValueError):
        bits("invalid")

    # Valid input
    b = bits("4bX1_0?")
    assert b._v.data == 0b11_10_01_00
    b = bits("64hFeDc_Ba98_7654_3210")
    assert b._v.data == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655
    b = bits("64hfEdC_bA98_7654_3210")
    assert b._v.data == 0xAAA9_A6A5_9A99_9695_6A69_6665_5A59_5655


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
    b = bits()

    # Test properties
    assert b.shape == (0,)
    assert b._v.data == 0
    assert b.ndim == 1
    assert b.size == 0
    assert b._v.nbits == 0
    assert list(b.flat) == []  # pylint: disable = use-implicit-booleaness-not-comparison

    assert b.flatten() == b

    # Test __str__ and __repr__
    assert str(b) == "bits([])"
    assert repr(b) == "bits((0,), 0b0)"

    # Test __len__
    assert len(b) == 0

    # Test __iter__
    assert list(b) == []  # pylint: disable = use-implicit-booleaness-not-comparison

    # Test __getitem__
    with pytest.raises(IndexError):
        _ = b[0]

    # Test __eq__
    assert b == bits()
    assert b == b.reshape((0,))
    assert b != bits(0)

    # Test to_uint, to_int
    assert b.to_uint() == 0
    assert b.to_int() == 0


def test_scalar():
    """Test scalar (vector w/ one element)."""
    bn = W
    b0 = bits(0)
    b1 = bits(1)
    bx = X

    # Test properties
    assert b0.shape == (1,)
    assert b0._v.data == 0b01
    assert b0.ndim == 1
    assert b0.size == 1
    assert b0._v.nbits == 2
    assert list(b0.flat) == [F]

    assert b0.flatten() == b0

    # Test __str__
    assert str(bn) == "bits([?])"
    assert str(b0) == "bits([0])"
    assert str(b1) == "bits([1])"
    assert str(bx) == "bits([X])"

    # Test __repr__
    assert repr(bn) == "bits((1,), 0b00)"
    assert repr(b0) == "bits((1,), 0b01)"
    assert repr(b1) == "bits((1,), 0b10)"
    assert repr(bx) == "bits((1,), 0b11)"

    # Test __len__
    assert len(b0) == 1

    # Test __iter__
    assert list(b0) == [F]

    # Test __getitem__
    assert b0[0] == F

    # Test __eq__
    assert b0 == bits(0)
    assert b0 == b0.reshape((1,))
    assert b0 != bits()
    assert b0 != bits(1)

    # Test to_uint, to_int
    with pytest.raises(ValueError):
        assert bn.to_uint()
    assert b0.to_uint() == 0
    assert b0.to_int() == 0
    assert b1.to_uint() == 1
    assert b1.to_int() == -1
    with pytest.raises(ValueError):
        assert bx.to_uint()


def test_rank1_str():
    """Test bits rank1 string input."""
    b = bits("8bX10?_X10?")
    data = 0b11100100_11100100
    xs = [W, F, T, X] * 2

    # Test properties
    assert b.shape == (8,)
    assert b._v.data == data
    assert b.ndim == 1
    assert b.size == 8
    assert b._v.nbits == 16
    assert list(b.flat) == xs

    assert b.flatten() == b

    # Test __str__ and __repr__
    assert str(b) == "bits(8bX10?_X10?)"
    assert repr(b) == "bits((8,), 0b1110_0100_1110_0100)"

    # Test __len__
    assert len(b) == 8

    # Test __iter__
    assert list(b) == xs

    # Test __getitem__
    assert b[0] == W
    assert b[1] == F
    assert b[6] == T
    assert b[7] == X

    # Test __eq__
    assert b == bits("8bX10?_X10?")
    # Same data, different shape
    assert b != b.reshape((2, 4))
    with pytest.raises(ValueError):
        b.reshape((42,))
    # Different data, same shape
    assert b != bits("8b0000_0000")

    # Test to_uint
    assert bits("16b1101_1110_1010_1101").to_uint() == 0xDEAD

    # Test to_int
    assert bits("4b0000").to_int() == 0
    assert bits("4b0001").to_int() == 1
    assert bits("4b0110").to_int() == 6
    assert bits("4b0111").to_int() == 7
    assert bits("4b1000").to_int() == -8
    assert bits("4b1001").to_int() == -7
    assert bits("4b1110").to_int() == -2
    assert bits("4b1111").to_int() == -1


def test_rank1_logic():
    """Test bits function w/ rank1 logic input."""
    xs = [W, F, T, X]
    b1 = bits("4bX10?")
    b2 = bits([0, 1, 0, 1])

    # Test properties
    assert b1.shape == (4,)
    assert b1._v.data == 0b11100100
    assert b1.ndim == 1
    assert b1.size == 4
    assert b1._v.nbits == 8
    assert list(b1.flat) == xs

    # Test __str__
    assert str(b1) == "bits(4bX10?)"
    assert str(b2) == "bits(4b1010)"

    # Test __repr__
    assert repr(b1) == "bits((4,), 0b1110_0100)"
    assert repr(b2) == "bits((4,), 0b1001_1001)"


def test_rank2_str():
    """Test bits function w/ rank2 str input."""
    b = bits(["4bX10?", "4bX10?"])

    assert b.flatten() == bits("8bX10?_X10?")

    # Test __str__
    assert str(b) == "bits([4bX10?, 4bX10?])"

    # Test __repr__
    assert repr(b) == "bits((2, 4), 0b1110_0100_1110_0100)"


def test_rank2_vec():
    """Test bits function w/ rank2 bits input."""
    b = bits([bits("4bX10?"), bits("4bX10?")])

    # Test __str__
    assert str(b) == "bits([4bX10?, 4bX10?])"

    # Test __repr__
    assert repr(b) == "bits((2, 4), 0b1110_0100_1110_0100)"


def test_rank2_errors():
    """Test bits function rank2 errors."""
    # Mismatched str literal
    with pytest.raises(TypeError):
        bits(["4bX10?", "3b10?"])
    # bits followed by some invalid type
    with pytest.raises(TypeError):
        bits(["4bX10?", 42])


R3VEC = """\
bits([[4bX10?, 4bX10?],
      [4bX10?, 4bX10?]])"""


def test_rank3_vec():
    """Test bits function w/ rank3 input."""
    b = bits(
        [
            [bits("4bX10?"), bits("4bX10?")],
            [bits("4bX10?"), bits("4bX10?")],
        ]
    )

    assert b.flatten() == bits("16bX10?_X10?_X10?_X10?")

    # Test __str__
    assert str(b) == R3VEC

    # Test __repr__
    assert repr(b) == "bits((2, 2, 4), 0b1110_0100_1110_0100_1110_0100_1110_0100)"


R4VEC = """\
bits([[[4bX10?, 4bX10?],
       [4bX10?, 4bX10?]],

      [[4bX10?, 4bX10?],
       [4bX10?, 4bX10?]]])"""


def test_rank4_vec():
    """Test bits function w/ rank4 input."""
    b = bits(
        [
            [[bits("4bX10?"), bits("4bX10?")], [bits("4bX10?"), bits("4bX10?")]],
            [[bits("4bX10?"), bits("4bX10?")], [bits("4bX10?"), bits("4bX10?")]],
        ]
    )

    # Test __str__
    assert str(b) == R4VEC

    # Test __repr__
    assert repr(b) == (
        "bits("
        "(2, 2, 2, 4), "
        "0b1110_0100_1110_0100_1110_0100_1110_0100_1110_0100_1110_0100_1110_0100_1110_0100"
        ")"
    )


def test_invalid_vec():
    """Test bits function invalid input."""
    with pytest.raises(TypeError):
        bits(42)


def test_cat():
    """Test bits cat function."""
    assert cat([]) == bits()
    assert cat([False, True, False, True]) == bits("4b1010")

    with pytest.raises(TypeError):
        cat(["invalid"])  # pyright: ignore[reportArgumentType]

    b = cat([bits("2b00"), bits("2b01"), bits("2b10"), bits("2b11")], flatten=True)
    assert b == bits("8b11100100")
    assert b.shape == (8,)

    b = cat([bits("2b00"), bits("2b01"), bits("2b10"), bits("2b11")], flatten=False)
    assert b.shape == (4, 2)

    b = cat([bits("1b0"), bits("2b01"), bits("1b1"), bits("2b11")], flatten=True)
    assert b.shape == (6,)

    b = cat([bits("1b0"), bits("2b01"), bits("1b1"), bits("2b11")], flatten=False)
    assert b.shape == (6,)

    # Incompatible shapes
    with pytest.raises(ValueError):
        cat([bits("2b00"), bits([bits("2b00"), bits("2b00")])])


def test_rep():
    """Test bits rep function."""
    b = rep(bits("2b00"), 4, flatten=True)
    assert b == bits("8b0000_0000")
    assert b.shape == (8,)

    b = rep(bits("2b00"), 4, flatten=False)
    assert b.shape == (4, 2)


def test_consts():
    """Test bits constants."""
    assert illogicals((8,)) == bits("8b????_????")
    assert zeros((8,)) == bits("8b0000_0000")
    assert ones((8,)) == bits("8b1111_1111")
    assert xes((8,)) == bits("8bXXXX_XXXX")


def test_slicing():
    """Test bits slicing behavior."""
    b = bits(
        [
            [bits("4b0000"), bits("4b0001"), bits("4b0010"), bits("4b0011")],
            [bits("4b0100"), bits("4b0101"), bits("4b0110"), bits("4b0111")],
            [bits("4b1000"), bits("4b1001"), bits("4b1010"), bits("4b1011")],
            [bits("4b1100"), bits("4b1101"), bits("4b1110"), bits("4b1111")],
        ]
    )

    assert b.shape == (4, 4, 4)

    with pytest.raises(IndexError):
        b[-5]
    with pytest.raises(TypeError):
        b["invalid"]  # pyright: ignore[reportArgumentType]

    assert b == b[:]
    assert b == b[0:4]
    assert b == b[-4:]
    assert b == b[-5:]
    assert b == b[:, :]
    assert b == b[:, :, :]

    assert b[0] == b[0, :]
    assert b[0] == b[0, 0:4]
    assert b[0] == b[0, -4:]
    assert b[0] == b[0, -5:]
    assert b[0] == b[0, :, :]

    assert b[0] == bits([bits("4b0000"), bits("4b0001"), bits("4b0010"), bits("4b0011")])
    assert b[1] == bits([bits("4b0100"), bits("4b0101"), bits("4b0110"), bits("4b0111")])
    assert b[2] == bits([bits("4b1000"), bits("4b1001"), bits("4b1010"), bits("4b1011")])
    assert b[3] == bits([bits("4b1100"), bits("4b1101"), bits("4b1110"), bits("4b1111")])

    assert b[0, 0] == b[0, 0, :]
    assert b[0, 0] == b[0, 0, 0:4]
    assert b[0, 0] == b[0, 0, -4:]
    assert b[0, 0] == b[0, 0, -5:]

    assert b[0, 0] == bits("4b0000")
    assert b[1, 1] == bits("4b0101")
    assert b[2, 2] == bits("4b1010")
    assert b[3, 3] == bits("4b1111")

    assert b[0, :, 0] == bits("4b1010")
    assert b[1, :, 1] == bits("4b1100")
    assert b[2, :, 2] == bits("4b0000")
    assert b[3, :, 3] == bits("4b1111")

    assert b[0, 0, :-1] == bits("3b000")
    assert b[0, 0, :-2] == bits("2b00")
    assert b[0, 0, :-3] == bits("1b0")
    assert b[0, 0, :-4] == bits()

    assert b[0, 0, 0] == F
    assert b[0, bits("2b00"), 0] == F
    assert b[-4, -4, -4] == F
    assert b[3, 3, 3] == T
    assert b[3, bits("2b11"), 3] == T
    assert b[-1, -1, -1] == T

    with pytest.raises(ValueError):
        b[0, 0, 0, 0]
    with pytest.raises(TypeError):
        b["invalid"]  # pyright: ignore[reportArgumentType]


def test_countbits():
    """Test bits countbits methods."""
    b = bits("8bX10?_X10?")
    assert b._v.count_illogicals() == 2
    assert b._v.count_zeros() == 2
    assert b._v.count_ones() == 2
    assert b._v.count_unknowns() == 2

    assert bits("4b0000")._v.count_ones() == 0
    assert bits("4b0001")._v.count_ones() == 1
    assert bits("4b0011")._v.count_ones() == 2
    assert bits("4b0111")._v.count_ones() == 3
    assert bits("4b1111")._v.count_ones() == 4

    assert not bits("4b0000")._v.onehot()
    assert bits("4b1000")._v.onehot()
    assert bits("4b0001")._v.onehot()
    assert not bits("4b1001")._v.onehot()
    assert not bits("4b1101")._v.onehot()

    assert bits("4b0000")._v.onehot0()
    assert bits("4b1000")._v.onehot0()
    assert not bits("4b1010")._v.onehot0()
    assert not bits("4b1011")._v.onehot0()

    assert not bits("4b0000")._v.has_unknown()
    assert not bits("4b1111")._v.has_unknown()
    assert bits("4b0X01")._v.has_unknown()
    assert bits("4b01?1")._v.has_illogical()
