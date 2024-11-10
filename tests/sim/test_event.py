"""Test seqlogic.sim.Event class."""

from seqlogic import Event, create_task, now, run, sleep


def log(s: str):
    print(f"{now():04} {s}")


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
0000 P1 enter
0000 S1 enter
0000 S1 waiting
0000 S2 enter
0000 S2 waiting
0000 S3 enter
0000 S3 waiting
0010 P1 set
0010 S1 running
0010 S2 running
0010 S3 running
0020 P1 exit
0020 S1 exit
0020 S2 exit
0020 S3 exit
"""


def test_acquire_release(capsys):

    async def main():
        event = Event()
        create_task(primary(event, "P1"))
        create_task(secondary(event, "S1"))
        create_task(secondary(event, "S2"))
        create_task(secondary(event, "S3"))

    run(main())

    assert capsys.readouterr().out == EXP1
