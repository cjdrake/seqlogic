"""Test seqlogic.lbool.VecStruct."""

# PyRight is confused by MetaClass behavior
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

from seqlogic.lbool import Vec, VecStruct, vec


class Pixel(VecStruct):
    r: Vec[8]
    g: Vec[8]
    b: Vec[8]


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
    assert repr(p) == 'Pixel(r=vec("8b0001_0001"), g=vec("8b0010_0010"), b=vec("8b0100_0100"))'

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
