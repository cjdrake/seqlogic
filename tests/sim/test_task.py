"""Test seqlogic.sim.Lock class."""

from seqlogic import CancelledError, Task, TaskGroup, create_task, now, run, sleep


def log(s: str):
    print(f"{now():04} {s}")


async def basic_c1():
    log("C1 enter")

    await sleep(10)
    log("C1")
    await sleep(10)
    log("C1")
    await sleep(10)
    log("C1")
    await sleep(10)

    log("C1 exit")


async def basic_c2(t: Task):
    log("C2 enter")

    log("C2 suspend")
    await t
    log("C2 resume")

    await t
    await sleep(10)
    await t

    log("C2 exit")


EXP1 = """\
0000 C1 enter
0000 C2 enter
0000 C2 suspend
0010 C1
0020 C1
0030 C1
0040 C1 exit
0040 C2 resume
0050 C2 exit
"""


def test_basic(capsys):

    async def main():
        t1 = create_task(basic_c1())
        create_task(basic_c2(t1))

    run(main())
    assert capsys.readouterr().out == EXP1


async def cancel_c1():
    log("C1 enter")

    try:
        await sleep(1000)
    except CancelledError:
        log("C1 except")
        raise
    finally:
        log("C1 finally")


async def cancel_c2():
    log("C2 enter")

    task = create_task(cancel_c1())

    await sleep(1)

    log("C2 cancels C1")
    task.cancel()

    try:
        await task
    except CancelledError:
        log("C2 except")
    finally:
        log("C2 finally")


EXP2 = """\
0000 C2 enter
0000 C1 enter
0001 C2 cancels C1
0001 C1 except
0001 C1 finally
0001 C2 except
0001 C2 finally
"""


def test_cancel(capsys):
    run(cancel_c2())
    assert capsys.readouterr().out == EXP2


async def group_c1():
    log("C1 enter")
    await sleep(5)
    log("C1 exit")
    return 1


async def group_c2():
    log("C2 enter")
    await sleep(10)
    log("C2 exit")
    return 2


async def group_c3():
    log("C3 enter")
    await sleep(15)
    log("C3 exit")
    return 3


EXP3 = """\
0000 C1 enter
0000 C2 enter
0000 C3 enter
0005 C1 exit
0010 C2 exit
0015 C3 exit
0015 MAIN
"""


def test_group(capsys):

    async def main():
        async with TaskGroup() as tg:
            t1 = tg.create_task(group_c1())
            t2 = tg.create_task(group_c2())
            t3 = tg.create_task(group_c3())

        log("MAIN")

        assert t1.result() == 1
        assert t2.result() == 2
        assert t3.result() == 3

    run(main())
    assert capsys.readouterr().out == EXP3
