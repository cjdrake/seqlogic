"""Test symbolic getitem / getattr"""

import pytest

from seqlogic import GetAttr
from seqlogic.expr import Variable

a = Variable(name="a")


def test_getitem():
    y = a[0]
    assert str(y) == "a[0]"

    y = a[:4]
    assert str(y) == "a[:4]"

    y = a[0:]
    assert str(y) == "a[0:]"

    y = a[1:3]
    assert str(y) == "a[1:3]"

    with pytest.raises(TypeError):
        _ = a[4.2]  # pyright: ignore[reportArgumentType]


def test_getattr():
    y = GetAttr(a, "foo")
    assert str(y) == "a.foo"

    with pytest.raises(TypeError):
        GetAttr(a, 4.2)  # pyright: ignore[reportArgumentType]
