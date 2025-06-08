"""Test Exceptions"""

import logging

import pytest
from deltacycle import run, sleep
from pytest import LogCaptureFixture

from seqlogic import Module

logger = logging.getLogger("deltacycle")


async def hello():
    logger.info("foo")
    await sleep(10)
    logger.info("bar")
    await sleep(10)
    raise ArithmeticError(42)


class Top(Module):
    def build(self):
        # Control
        self.drv(hello())


EXP1 = {
    (0, "foo"),
    (10, "bar"),
}


def test_hello(caplog: LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="deltacycle")

    top = Top(name="top")
    with pytest.raises(ExceptionGroup) as e:
        run(top.main())
    # excs = e.value.args[1]
    assert len(e.value.exceptions) == 1
    assert isinstance(e.value.exceptions[0], ArithmeticError)
    assert e.value.exceptions[0].args == (42,)

    msgs = {(r.time, r.getMessage()) for r in caplog.records}
    assert msgs == EXP1
