"""
Test seqlogic.sim
"""

from collections import defaultdict

from seqlogic.sim import SimVar, get_loop, notify, sleep

loop = get_loop()
waves = defaultdict(dict)


def waves_add(time, var, val):
    waves[time][var] = val


class TraceVar(SimVar):
    """
    Variable that supports dumping to memory.
    """

    def __init__(self):
        super().__init__(value=False)
        waves_add(self._sim.time(), self, self._value)

    def update(self):
        if self.dirty():
            waves_add(self._sim.time(), self, self._next)
        super().update()

    def edge(self):
        return not self._value and self._next or self._value and not self._next


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

    loop.add_proc(hello, 0)

    loop.run(42)

    assert capsys.readouterr().out == HELLO_OUT


def test_vars():
    """Test generic variable functionality."""
    loop.reset()

    a = TraceVar()
    b = TraceVar()

    async def run_a():
        while True:
            await sleep(5)
            a.next = not a.value

    async def run_b():
        i = 0
        while True:
            await notify(a.edge)
            if i % 2 == 0:
                b.next = not b.value
            else:
                b.next = b.value
            i += 1

    loop.add_proc(run_a, 1)
    loop.add_proc(run_b, 0)

    loop.run(50)

    exp = {
        -1: {
            a: False,
            b: False,
        },
        5: {
            a: True,
            b: True,
        },
        10: {
            a: False,
        },
        15: {
            a: True,
            b: False,
        },
        20: {
            a: False,
        },
        25: {
            a: True,
            b: True,
        },
        30: {
            a: False,
        },
        35: {
            a: True,
            b: False,
        },
        40: {
            a: False,
        },
        45: {
            a: True,
            b: True,
        },
    }

    assert waves == exp
