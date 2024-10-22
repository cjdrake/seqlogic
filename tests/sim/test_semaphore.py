"""Test seqlogic.sim.Semaphore class."""

from seqlogic import Semaphore, create_task, now, run, sleep


def log(s: str):
    print(f"{now():04} {s}")


async def foo(sem: Semaphore, name: str, t1: int, t2: int):
    log(f"{name} enter")

    await sleep(t1)

    log(f"{name} attempt acquire")
    await sem.acquire()
    log(f"{name} acquired")

    try:
        await sleep(t2)
    finally:
        log(f"{name} release")
        sem.release()

    await sleep(10)
    log(f"{name} exit")


async def bar(sem: Semaphore, name: str, t1: int, t2: int):
    log(f"{name} enter")

    await sleep(t1)

    log(f"{name} attempt acquire")
    async with sem:
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
0000 4 enter
0000 5 enter
0000 6 enter
0000 7 enter
0010 0 attempt acquire
0010 0 acquired
0011 1 attempt acquire
0011 1 acquired
0012 2 attempt acquire
0012 2 acquired
0013 3 attempt acquire
0013 3 acquired
0014 4 attempt acquire
0015 5 attempt acquire
0016 6 attempt acquire
0017 7 attempt acquire
0020 0 release
0020 4 acquired
0021 1 release
0021 5 acquired
0022 2 release
0022 6 acquired
0023 3 release
0023 7 acquired
0030 0 exit
0030 4 release
0031 1 exit
0031 5 release
0032 2 exit
0032 6 release
0033 3 exit
0033 7 release
0040 4 exit
0041 5 exit
0042 6 exit
0043 7 exit
"""


def test_acquire_release(capsys):

    async def main():
        sem = Semaphore(4)
        for i in range(8):
            create_task(foo(sem, f"{i}", i + 10, 10))

    run(main())

    assert capsys.readouterr().out == EXP


def test_async_with(capsys):

    async def main():
        sem = Semaphore(4)
        for i in range(8):
            create_task(bar(sem, f"{i}", i + 10, 10))

    run(main())

    assert capsys.readouterr().out == EXP
