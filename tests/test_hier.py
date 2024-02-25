"""Test hierarchy module."""

from seqlogic import Module
from seqlogic.hier import Dict, HierVar, List


def test_basic():
    """Test basic hierarchical naming."""
    a = Module(name="a")
    w = HierVar(name="w", parent=a)
    b = Module(name="b", parent=a)
    x = HierVar(name="x", parent=b)
    y = HierVar(name="y", parent=b)

    assert a.name == "a"
    assert a.qualname == "/a"
    assert b.name == "b"
    assert b.qualname == "/a/b"

    assert w.name == "w"
    assert w.qualname == "/a/w"
    assert x.name == "x"
    assert x.qualname == "/a/b/x"
    assert y.name == "y"
    assert y.qualname == "/a/b/y"

    assert list(a.iter_bfs()) == [a, w, b, x, y]
    assert list(a.iter_dfs()) == [w, x, y, b, a]


def test_list():
    """Test list naming."""
    top = Module(name="top")

    a = List(name="l", parent=top)
    a.append(HierVar(name="0", parent=a))
    a.append(Module(name="1", parent=a))

    assert a.name == "l"
    assert a[0].name == "0"
    assert a[0].qualname == "/top/l[0]"
    assert a[1].name == "1"
    assert a[1].qualname == "/top/l[1]"


def test_dict():
    """Test dict naming."""
    top = Module(name="top")

    a = Dict(name="a", parent=top)
    a["w"] = HierVar(name="w", parent=a)
    a["b"] = Module(name="b", parent=a)

    assert a.name == "a"
    assert a["w"].name == "w"
    assert a["w"].qualname == "/top/a[w]"
    assert a["b"].name == "b"
    assert a["b"].qualname == "/top/a[b]"
