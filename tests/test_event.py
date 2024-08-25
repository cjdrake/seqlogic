"""Test seqlogic.sim.Event class."""

from seqlogic import get_loop, sleep
from seqlogic.sim import Event

loop = get_loop()


def log(s: str):
    print(f"{loop.time():04} {s}")


async def foo(event: Event):
    log("FOO enter")
    await sleep(10)
    log("FOO set")
    event.set()
    assert event.is_set()
    await sleep(10)
    log("FOO exit")


async def bar(event: Event):
    log("BAR enter")
    log("BAR waiting")
    await event.wait()
    log("BAR running")
    await sleep(10)
    log("BAR exit")


async def fiz(event: Event):
    log("FIZ enter")
    log("FIZ waiting")
    await event.wait()
    log("FIZ running")
    await sleep(10)
    log("FIZ exit")


async def buz(event: Event):
    log("BUZ enter")
    log("BUZ waiting")
    await event.wait()
    log("BUZ running")
    await sleep(10)
    log("BUZ exit")


EXP1 = """\
0000 FOO enter
0000 BAR enter
0000 BAR waiting
0000 FIZ enter
0000 FIZ waiting
0000 BUZ enter
0000 BUZ waiting
0010 FOO set
0010 BAR running
0010 FIZ running
0010 BUZ running
0020 FOO exit
0020 BAR exit
0020 FIZ exit
0020 BUZ exit
"""


def test_acquire_release(capsys):
    loop.reset()

    event = Event()
    loop.add_active(foo(event))
    loop.add_active(bar(event))
    loop.add_active(fiz(event))
    loop.add_active(buz(event))
    loop.run()

    assert capsys.readouterr().out == EXP1
