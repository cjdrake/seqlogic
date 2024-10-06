"""Test seqlogic.sim.Lock class."""

from seqlogic import create_task, now, run, sleep
from seqlogic.sim import Lock


def log(s: str):
    print(f"{now():04} {s}")


async def foo(lock: Lock):
    log("FOO enter")

    await sleep(10)

    log("FOO attempt acquire")
    await lock.acquire()
    log("FOO acquired")

    try:
        await sleep(10)
    finally:
        log("FOO release")
        lock.release()

    await sleep(10)
    log("FOO exit")


async def bar(lock: Lock):
    log("BAR enter")

    await sleep(15)

    log("BAR attempt acquire")
    await lock.acquire()
    log("BAR acquired")

    try:
        await sleep(10)
    finally:
        log("BAR release")
        lock.release()

    log("BAR exit")


EXP1 = """\
0000 FOO enter
0000 BAR enter
0010 FOO attempt acquire
0010 FOO acquired
0015 BAR attempt acquire
0020 FOO release
0020 BAR acquired
0030 FOO exit
0030 BAR release
0030 BAR exit
"""


def test_acquire_release(capsys):

    async def main():
        lock = Lock()
        create_task(foo(lock))
        create_task(bar(lock))

    run(main())

    assert capsys.readouterr().out == EXP1


async def fiz(lock: Lock):
    log("FIZ enter")

    await sleep(10)

    log("FIZ attempt acquire")
    async with lock:
        log("FIZ acquired")
        await sleep(10)
    log("FIZ release")

    await sleep(10)
    log("FIZ exit")


async def buz(lock: Lock):
    log("BUZ enter")

    await sleep(15)

    log("BUZ attempt acquire")
    async with lock:
        log("BUZ acquired")
        await sleep(10)
    log("BUZ release")

    log("BUZ exit")


EXP2 = """\
0000 FIZ enter
0000 BUZ enter
0010 FIZ attempt acquire
0010 FIZ acquired
0015 BUZ attempt acquire
0020 FIZ release
0020 BUZ acquired
0030 FIZ exit
0030 BUZ release
0030 BUZ exit
"""


def test_async_with(capsys):

    async def main():
        lock = Lock()
        create_task(fiz(lock))
        create_task(buz(lock))

    run(main())

    assert capsys.readouterr().out == EXP2
