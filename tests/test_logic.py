"""
Test Logic Data Type
"""


from seqlogic.logic import logic

char2logic = {
    "N": logic.N,
    "0": logic.F,
    "1": logic.T,
    "X": logic.X,
}


def test_str():
    """Test logic.__str__ method"""
    assert str(logic.N) == "X"
    assert str(logic.F) == "0"
    assert str(logic.T) == "1"
    assert str(logic.X) == "x"


def test_repr():
    """Test logic.__repr__ method"""
    assert repr(logic.N) == "X"
    assert repr(logic.F) == "0"
    assert repr(logic.T) == "1"
    assert repr(logic.X) == "x"


def test_not():
    """Test logic NOT function"""
    table = {
        "0": "1",
        "1": "0",
        "N": "N",
        "X": "X",
    }
    for x, y in table.items():
        x = char2logic[x]
        y = char2logic[y]
        assert ~x == y


def test_nor():
    """Test logic NOR function"""
    table = {
        "00": "1",
        "01": "0",
        "10": "0",
        "11": "0",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "X",
        "1X": "0",
        "X0": "X",
        "X1": "0",
        "XX": "X",
    }
    for xs, y in table.items():
        x0 = char2logic[xs[0]]
        x1 = char2logic[xs[1]]
        y = char2logic[y]
        assert x0.lnor(x1) == y
        assert ~(x0 | x1) == y

    assert logic.F.lnor(False) is logic.T
    assert logic.F.lnor(True) is logic.F
    assert logic.F.lnor(0) is logic.T
    assert logic.F.lnor(1) is logic.F


def test_or():
    """Test logic OR function"""
    table = {
        "00": "0",
        "01": "1",
        "10": "1",
        "11": "1",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "X",
        "1X": "1",
        "X0": "X",
        "X1": "1",
        "XX": "X",
    }
    for xs, y in table.items():
        x0 = char2logic[xs[0]]
        x1 = char2logic[xs[1]]
        y = char2logic[y]
        assert x0.lor(x1) == y
        assert (x0 | x1) == y

    assert logic.F | False is logic.F
    assert logic.F | True is logic.T
    assert logic.F | 0 is logic.F
    assert logic.F | 1 is logic.T

    assert False | logic.F is logic.F
    assert True | logic.F is logic.T
    assert 0 | logic.F is logic.F
    assert 1 | logic.F is logic.T


def test_nand():
    """Test logic NAND function"""
    table = {
        "00": "1",
        "01": "1",
        "10": "1",
        "11": "0",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "1",
        "1X": "X",
        "X0": "1",
        "X1": "X",
        "XX": "X",
    }
    for xs, y in table.items():
        x0 = char2logic[xs[0]]
        x1 = char2logic[xs[1]]
        y = char2logic[y]
        assert x0.lnand(x1) == y
        assert ~(x0 & x1) == y

    assert logic.T.lnand(False) is logic.T
    assert logic.T.lnand(True) is logic.F
    assert logic.T.lnand(0) is logic.T
    assert logic.T.lnand(1) is logic.F


def test_and():
    """Test logic AND function"""
    table = {
        "00": "0",
        "01": "0",
        "10": "0",
        "11": "1",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "0",
        "1X": "X",
        "X0": "0",
        "X1": "X",
        "XX": "X",
    }
    for xs, y in table.items():
        x0 = char2logic[xs[0]]
        x1 = char2logic[xs[1]]
        y = char2logic[y]
        assert x0.land(x1) == y
        assert (x0 & x1) == y

    assert logic.T & False is logic.F
    assert logic.T & True is logic.T
    assert logic.T & 0 is logic.F
    assert logic.T & 1 is logic.T

    assert False & logic.T is logic.F
    assert True & logic.T is logic.T
    assert 0 & logic.T is logic.F
    assert 1 & logic.T is logic.T


def test_xnor():
    """Test logic XNOR function"""
    table = {
        "00": "1",
        "01": "0",
        "10": "0",
        "11": "1",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "X",
        "1X": "X",
        "X0": "X",
        "X1": "X",
        "XX": "X",
    }
    for xs, y in table.items():
        x0 = char2logic[xs[0]]
        x1 = char2logic[xs[1]]
        y = char2logic[y]
        assert x0.lxnor(x1) == y
        assert ~(x0 ^ x1) == y

    assert logic.F.lxnor(False) is logic.T
    assert logic.F.lxnor(True) is logic.F
    assert logic.F.lxnor(0) is logic.T
    assert logic.F.lxnor(1) is logic.F


def test_xor():
    """Test logic XOR function"""
    table = {
        "00": "0",
        "01": "1",
        "10": "1",
        "11": "0",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "X",
        "1X": "X",
        "X0": "X",
        "X1": "X",
        "XX": "X",
    }
    for xs, y in table.items():
        x0 = char2logic[xs[0]]
        x1 = char2logic[xs[1]]
        y = char2logic[y]
        assert x0.lxor(x1) == y
        assert (x0 ^ x1) == y

    assert logic.F ^ False is logic.F
    assert logic.F ^ True is logic.T
    assert logic.F ^ 0 is logic.F
    assert logic.F ^ 1 is logic.T

    assert False ^ logic.F is logic.F
    assert True ^ logic.F is logic.T
    assert 0 ^ logic.F is logic.F
    assert 1 ^ logic.F is logic.T


def test_implies():
    """Test logic Implies function"""
    table = {
        "00": "1",
        "01": "1",
        "10": "0",
        "11": "1",
        "0N": "N",
        "1N": "N",
        "N0": "N",
        "N1": "N",
        "XN": "N",
        "NX": "N",
        "NN": "N",
        "0X": "1",
        "1X": "X",
        "X0": "X",
        "X1": "1",
        "XX": "X",
    }
    for xs, y in table.items():
        p = char2logic[xs[0]]
        q = char2logic[xs[1]]
        y = char2logic[y]
        assert p.limplies(q) == y
        assert (~p | q) == p.limplies(q)

    assert logic.T.limplies(False) is logic.F
    assert logic.T.limplies(True) is logic.T
    assert logic.T.limplies(0) is logic.F
    assert logic.T.limplies(1) is logic.T
