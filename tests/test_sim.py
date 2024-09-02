"""Test seqlogic.sim module."""

from collections import defaultdict

import pytest

from seqlogic import get_loop, resume, sleep
from seqlogic.sim import Singular, State

loop = get_loop()
waves = defaultdict(dict)


def _waves_add(time, var, val):
    waves[time][var] = val


class Bool(Singular):
    """Variable that supports dumping to memory."""

    def __init__(self):
        super().__init__(value=False)
        _waves_add(self._sim.time(), self, self._value)

    def update(self):
        if self.dirty():
            _waves_add(self._sim.time(), self, self._next_value)
        super().update()

    def is_posedge(self) -> bool:
        return not self._value and self._next_value

    def is_negedge(self) -> bool:
        return self._value and not self._next_value

    def is_edge(self) -> bool:
        return self.is_posedge() or self.is_negedge()

    async def posedge(self) -> State:
        await resume((self, self.is_posedge))

    async def negedge(self) -> State:
        await resume((self, self.is_negedge))

    async def edge(self) -> State:
        await resume((self, self.is_edge))


HELLO_OUT = """\
[2] Hello
[4] World
"""


def test_hello(capsys):
    """Test basic async/await hello world functionality."""
    loop.reset()

    async def hello():
        await sleep(2)
        print(f"[{loop.time()}] Hello")
        await sleep(2)
        print(f"[{loop.time()}] World")

    loop.add_initial(hello())

    # Invalid run limit
    with pytest.raises(TypeError):
        loop.run("Invalid argument type")

    # Run until no events left
    loop.run()

    assert capsys.readouterr().out == HELLO_OUT


def test_vars_run():
    """Test generic variable functionality."""
    waves.clear()
    loop.reset()

    clk = Bool()
    a = Bool()
    b = Bool()

    async def p_clk():
        while True:
            await sleep(5)
            clk.next = not clk.value

    async def p_a():
        i = 0
        while True:
            await clk.edge()
            if i % 2 == 0:
                a.next = not a.value
            else:
                a.next = a.value
            i += 1

    async def p_b():
        i = 0
        while True:
            await clk.edge()
            if i % 3 == 0:
                b.next = not b.value
            i += 1

    loop.add_initial(p_clk())
    loop.add_initial(p_a())
    loop.add_initial(p_b())

    # Expected sim output
    exp = {
        -1: {clk: False, a: False, b: False},
        5: {clk: True, a: True, b: True},
        10: {clk: False},
        15: {clk: True, a: False},
        20: {clk: False, b: False},
        25: {clk: True, a: True},
        30: {clk: False},
        35: {clk: True, a: False, b: True},
        40: {clk: False},
        45: {clk: True, a: True},
    }

    # Relative run limit
    loop.run(ticks=25)

    # Absolute run limit
    loop.run(until=50)

    assert waves == exp


def test_vars_iter():
    """Test generic variable functionality."""
    waves.clear()
    loop.reset()

    clk = Bool()
    a = Bool()
    b = Bool()

    async def p_clk():
        while True:
            await sleep(5)
            clk.next = not clk.value

    async def p_a():
        i = 0
        while True:
            await clk.edge()
            if i % 2 == 0:
                a.next = not a.value
            else:
                a.next = a.value
            i += 1

    async def p_b():
        i = 0
        while True:
            await clk.edge()
            if i % 3 == 0:
                b.next = not b.value
            i += 1

    loop.add_initial(p_clk())
    loop.add_initial(p_a())
    loop.add_initial(p_b())

    # Expected sim output
    exp = {
        -1: {clk: False, a: False, b: False},
        5: {clk: True, a: True, b: True},
        10: {clk: False},
        15: {clk: True, a: False},
        20: {clk: False, b: False},
        25: {clk: True, a: True},
        30: {clk: False},
        35: {clk: True, a: False, b: True},
        40: {clk: False},
        45: {clk: True, a: True},
    }

    for _ in loop.iter(until=50):
        pass

    assert waves == exp
