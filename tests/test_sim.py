"""Test seqlogic.sim module."""

from collections import defaultdict

import pytest

from seqlogic.sim import Region, SimVar, get_loop, notify, sleep

loop = get_loop()
waves = defaultdict(dict)


def _waves_add(time, var, val):
    waves[time][var] = val


class _BoolVar(SimVar):
    """Variable that supports dumping to memory."""

    def __init__(self):
        super().__init__(value=False)
        _waves_add(self._sim.time(), self, self._value)

    def update(self):
        if self.dirty():
            _waves_add(self._sim.time(), self, self._next_value)
        super().update()

    def edge(self) -> bool:
        return not self._value and self._next_value or self._value and not self._next_value


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

    loop.add_proc(hello, Region(0))

    # Invalid run limit
    with pytest.raises(TypeError):
        loop.run("Invalid argument type")  # pyright: ignore[reportArgumentType]

    # Run until no events left
    loop.run()

    assert capsys.readouterr().out == HELLO_OUT


def test_vars_run():
    """Test generic variable functionality."""
    waves.clear()
    loop.reset()

    a = _BoolVar()
    b = _BoolVar()
    c = _BoolVar()

    async def run_a():
        while True:
            await sleep(5)
            a.next = not a.value

    async def run_b():
        i = 0
        while True:
            await notify(a.edge)
            # Dirty next state
            if i % 2 == 0:
                b.next = not b.value
            # Clean next state
            else:
                b.next = b.value
            i += 1

    async def run_c():
        i = 0
        while True:
            await notify(a.edge)
            if i % 3 == 0:
                c.next = not c.value
            i += 1

    loop.add_proc(run_a, Region(1))
    loop.add_proc(run_b, Region(0))
    loop.add_proc(run_c, Region(0))

    # Expected sim output
    exp = {
        -1: {a: False, b: False, c: False},
        5: {a: True, b: True, c: True},
        10: {a: False},
        15: {a: True, b: False},
        20: {a: False, c: False},
        25: {a: True, b: True},
        30: {a: False},
        35: {a: True, b: False, c: True},
        40: {a: False},
        45: {a: True, b: True},
    }

    # Event loop not started yet
    assert not loop.started

    # Relative run limit
    loop.run(ticks=25)
    assert loop.started

    # Absolute run limit
    loop.run(until=50)

    assert waves == exp


def test_vars_iter():
    """Test generic variable functionality."""
    waves.clear()
    loop.reset()

    a = _BoolVar()
    b = _BoolVar()
    c = _BoolVar()

    async def run_a():
        while True:
            await sleep(5)
            a.next = not a.value

    async def run_b():
        i = 0
        while True:
            await notify(a.edge)
            # Dirty next state
            if i % 2 == 0:
                b.next = not b.value
            # Clean next state
            else:
                b.next = b.value
            i += 1

    async def run_c():
        i = 0
        while True:
            await notify(a.edge)
            if i % 3 == 0:
                c.next = not c.value
            i += 1

    loop.add_proc(run_a, Region(1))
    loop.add_proc(run_b, Region(0))
    loop.add_proc(run_c, Region(0))

    # Expected sim output
    exp = {
        -1: {a: False, b: False, c: False},
        5: {a: True, b: True, c: True},
        10: {a: False},
        15: {a: True, b: False},
        20: {a: False, c: False},
        25: {a: True, b: True},
        30: {a: False},
        35: {a: True, b: False, c: True},
        40: {a: False},
        45: {a: True, b: True},
    }

    for _ in loop.iter(until=50):
        pass

    assert waves == exp
