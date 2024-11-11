"""Test seqlogic.sim.Lock class."""

from seqlogic import Lock, create_task, now, run, sleep


def log(s: str):
    print(f"{now():04} {s}")


async def use_acquire_release(lock: Lock, name: str, t1: int, t2: int):
    log(f"{name} enter")

    await sleep(t1)

    log(f"{name} attempt acquire")
    await lock.acquire()
    log(f"{name} acquired")

    try:
        await sleep(t2)
    finally:
        log(f"{name} release")
        lock.release()

    await sleep(10)
    log(f"{name} exit")


async def use_with(lock: Lock, name: str, t1: int, t2: int):
    log(f"{name} enter")

    await sleep(t1)

    log(f"{name} attempt acquire")
    async with lock:
        log(f"{name} acquired")
        await sleep(t2)
    log(f"{name} release")

    await sleep(10)
    log(f"{name} exit")


EXP = """\
0000 0 enter
0000 1 enter
0000 2 enter
0000 3 enter
0010 0 attempt acquire
0010 0 acquired
0011 1 attempt acquire
0012 2 attempt acquire
0013 3 attempt acquire
0020 0 release
0020 1 acquired
0030 0 exit
0030 1 release
0030 2 acquired
0040 1 exit
0040 2 release
0040 3 acquired
0050 2 exit
0050 3 release
0060 3 exit
"""


def test_acquire_release(capsys):

    async def main():
        lock = Lock()
        for i in range(4):
            create_task(use_acquire_release(lock, f"{i}", i + 10, 10))

    run(main())

    assert capsys.readouterr().out == EXP


def test_async_with(capsys):

    async def main():
        lock = Lock()
        for i in range(4):
            create_task(use_with(lock, f"{i}", i + 10, 10))

    run(main())

    assert capsys.readouterr().out == EXP
