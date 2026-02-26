"""Test Exceptions"""

import pytest
from deltacycle import run, sleep

from seqlogic import Module

from .conftest import trace


async def hello():
    trace("foo")
    await sleep(10)
    trace("bar")
    await sleep(10)
    raise ArithmeticError(42)


class Top(Module):
    def build(self):
        # Control
        self.drv(hello)


EXP1 = {
    (0, "Task-0", "foo"),
    (10, "Task-0", "bar"),
}


def test_hello(captrace: set[tuple[int, str, str]]):
    top = Top(name="top")
    with pytest.raises(ExceptionGroup) as e:
        run(top.main())
    # excs = e.value.args[1]

    assert len(e.value.exceptions) == 1
    assert isinstance(e.value.exceptions[0], ArithmeticError)
    assert e.value.exceptions[0].args == (42,)

    assert captrace == EXP1
