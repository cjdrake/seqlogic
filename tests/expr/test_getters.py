"""Test symbolic getitem / getattr"""

import pytest

from seqlogic import GetAttr, GetItem, Variable

a = Variable(name="a")


def test_getitem():
    y = GetItem(a, 0)
    assert str(y) == "a[0]"

    y = GetItem(a, slice(None, 4))
    assert str(y) == "a[:4]"

    y = GetItem(a, slice(0, None))
    assert str(y) == "a[0:]"

    y = GetItem(a, slice(1, 3))
    assert str(y) == "a[1:3]"

    with pytest.raises(TypeError):
        GetItem(a, 4.2)


def test_getattr():
    y = GetAttr(a, "foo")
    assert str(y) == "a.foo"

    with pytest.raises(TypeError):
        GetAttr(a, 4.2)
