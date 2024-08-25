"""Test seqlogic.sim.Lock class."""

from seqlogic import get_loop, sleep
from seqlogic.sim import Task, create_task

loop = get_loop()


def log(s: str):
    print(f"{loop.time():04} {s}")


async def foo():
    log("FOO enter")

    await sleep(10)
    log("FOO")
    await sleep(10)
    log("FOO")
    await sleep(10)
    log("FOO")
    await sleep(10)

    log("FOO exit")


async def bar(t: Task):
    log("BAR enter")

    log("BAR suspend")
    await t
    log("BAR resume")

    await t
    await sleep(10)
    await t

    log("BAR exit")


async def main():
    t1 = create_task(foo())
    create_task(bar(t1))


EXP1 = """\
0000 FOO enter
0000 BAR enter
0000 BAR suspend
0010 FOO
0020 FOO
0030 FOO
0040 FOO exit
0040 BAR resume
0050 BAR exit
"""


def test_basic(capsys):
    loop.reset()

    loop.add_active(main())
    loop.run()

    assert capsys.readouterr().out == EXP1
