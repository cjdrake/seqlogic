"""Test seqlogic.lbool.VecStruct."""

# PyRight is confused by MetaClass behavior
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

import pytest

from seqlogic.lbool import Vec, VecEnum, VecStruct, vec


class Pixel(VecStruct):
    r: Vec[8]
    g: Vec[8]
    b: Vec[8]


class MyEnum(VecEnum):
    A = "2b01"
    B = "2b10"


class Mixed(VecStruct):
    a: Vec[4]
    b: MyEnum


class Nested(VecStruct):
    q: Vec[4]
    r: MyEnum
    s: Mixed


def test_empty():
    class EmptyStruct(VecStruct):
        pass

    s = EmptyStruct()
    assert len(s) == 0
    assert s.data == 0
    assert str(s) == "EmptyStruct()"


def test_basic():
    p = Pixel(r=vec("8b0001_0001"), g=vec("8b0010_0010"), b=vec("8b0100_0100"))

    assert str(p.r) == "8b0001_0001"
    assert str(p.g) == "8b0010_0010"
    assert str(p.b) == "8b0100_0100"

    assert str(p) == "Pixel(r=8b0001_0001, g=8b0010_0010, b=8b0100_0100)"
    assert repr(p) == (
        "Pixel("
        "r=Vec[8](0b0101011001010110)"
        ", g=Vec[8](0b0101100101011001)"
        ", b=Vec[8](0b0110010101100101))"
    )

    assert len(p) == 24

    # Partial assignment
    p = Pixel()
    assert str(p) == "Pixel(r=8bXXXX_XXXX, g=8bXXXX_XXXX, b=8bXXXX_XXXX)"
    p = Pixel(r=vec("8b0001_0001"))
    assert str(p) == "Pixel(r=8b0001_0001, g=8bXXXX_XXXX, b=8bXXXX_XXXX)"
    p = Pixel(g=vec("8b0010_0010"))
    assert str(p) == "Pixel(r=8bXXXX_XXXX, g=8b0010_0010, b=8bXXXX_XXXX)"
    p = Pixel(b=vec("8b0100_0100"))
    assert str(p) == "Pixel(r=8bXXXX_XXXX, g=8bXXXX_XXXX, b=8b0100_0100)"
    p = Pixel(r=vec("8b0001_0001"), g=vec("8b0010_0010"))
    assert str(p) == "Pixel(r=8b0001_0001, g=8b0010_0010, b=8bXXXX_XXXX)"
    p = Pixel(r=vec("8b0001_0001"), b=vec("8b0100_0100"))
    assert str(p) == "Pixel(r=8b0001_0001, g=8bXXXX_XXXX, b=8b0100_0100)"
    p = Pixel(g=vec("8b0010_0010"), b=vec("8b0100_0100"))
    assert str(p) == "Pixel(r=8bXXXX_XXXX, g=8b0010_0010, b=8b0100_0100)"

    assert str(Pixel.xes()) == "Pixel(r=8bXXXX_XXXX, g=8bXXXX_XXXX, b=8bXXXX_XXXX)"
    assert str(Pixel.dcs()) == "Pixel(r=8b----_----, g=8b----_----, b=8b----_----)"


def test_mixed():
    mx = Mixed()
    assert str(mx) == "Mixed(a=4bXXXX, b=MyEnum.X)"
    assert repr(mx) == "Mixed(a=Vec[4](0b00000000), b=MyEnum.X)"

    m1 = Mixed(a=vec("4b1010"), b=MyEnum.A)
    assert str(m1) == "Mixed(a=4b1010, b=MyEnum.A)"
    assert m1.b.B.name == "B"


def test_nested():
    n1 = Nested()
    assert str(n1) == "Nested(q=4bXXXX, r=MyEnum.X, s=Mixed(a=4bXXXX, b=MyEnum.X))"

    n2 = Nested(q=vec("4b1010"), r=MyEnum.A, s=Mixed(a=vec("4b0101"), b=MyEnum.B))
    assert str(n2) == "Nested(q=4b1010, r=MyEnum.A, s=Mixed(a=4b0101, b=MyEnum.B))"


def test_init_errors():
    with pytest.raises(TypeError):
        Pixel(r=vec("7b010_1010"))

    with pytest.raises(TypeError):
        Pixel(r=vec("9b0_0000_0000"))
