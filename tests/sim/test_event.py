"""Test seqlogic.sim.Event class."""

from seqlogic import get_loop, sleep
from seqlogic.sim import Event

loop = get_loop()


def log(s: str):
    print(f"{loop.time():04} {s}")


async def primary(event: Event, name: str):
    log(f"{name} enter")
    await sleep(10)
    log(f"{name} set")
    event.set()
    assert event.is_set()
    await sleep(10)
    log(f"{name} exit")


async def secondary(event: Event, name: str):
    log(f"{name} enter")
    log(f"{name} waiting")
    await event.wait()
    log(f"{name} running")
    await sleep(10)
    log(f"{name} exit")


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
    loop.add_initial(primary(event, "FOO"))
    loop.add_initial(secondary(event, "BAR"))
    loop.add_initial(secondary(event, "FIZ"))
    loop.add_initial(secondary(event, "BUZ"))
    loop.run()

    assert capsys.readouterr().out == EXP1
