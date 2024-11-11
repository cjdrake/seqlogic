"""Test seqlogic.sim.wait function."""

import pytest

from seqlogic import create_task, now, run, sleep, wait


def log(s: str):
    print(f"{now():04} {s}")


async def c(i: int, t: int):
    log(f"C{i} enter")
    await sleep(t)
    log(f"C{i} exit")


EXP1 = """\
0000 MAIN enter
0000 C1 enter
0000 C2 enter
0000 C3 enter
0005 C1 exit
0005 MAIN wait done
0010 C2 exit
0015 C3 exit
0015 MAIN exit
"""


def test_first(capsys):

    async def main():
        log("MAIN enter")

        t1 = create_task(c(1, 5))
        t2 = create_task(c(2, 10))
        t3 = create_task(c(3, 15))

        done, pend = await wait([t1, t2, t3], return_when="FIRST_COMPLETED")
        assert done == {t1}
        assert pend == {t2, t3}

        log("MAIN wait done")

        await t2
        await t3

        log("MAIN exit")

    run(main())
    assert capsys.readouterr().out == EXP1


EXP3 = """\
0000 MAIN enter
0000 C1 enter
0000 C2 enter
0000 C3 enter
0005 C1 exit
0010 C2 exit
0015 C3 exit
0015 MAIN exit
"""


def test_all(capsys):

    async def main():
        log("MAIN enter")

        t1 = create_task(c(1, 5))
        t2 = create_task(c(2, 10))
        t3 = create_task(c(3, 15))

        done, pend = await wait([t1, t2, t3], return_when="ALL_COMPLETED")
        assert done == {t1, t2, t3}
        assert not pend

        log("MAIN exit")

    run(main())
    assert capsys.readouterr().out == EXP3


def test_error1():
    async def main():
        await wait([], return_when="invalid")

    with pytest.raises(ValueError):
        run(main())


def test_error2():
    async def main():
        await wait([], return_when="FIRST_EXCEPTION")

    with pytest.raises(NotImplementedError):
        run(main())
