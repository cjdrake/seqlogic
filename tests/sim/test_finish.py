"""Test seqlogic.sim finish."""

from seqlogic import finish, get_loop, sleep

loop = get_loop()


def log(s: str):
    print(f"{loop.time():04} {s}")


async def main():
    log("MAIN enter")
    await sleep(100)

    # Force all PING threads to stop immediately
    log("MAIN finish")
    finish()


async def ping(name: str, period: int):
    log(f"{name} enter")
    while True:
        await sleep(period)
        log(f"{name} PING")


EXP1 = """\
0000 MAIN enter
0000 FOO enter
0000 BAR enter
0000 FIZ enter
0000 BUZ enter
0003 FOO PING
0005 BAR PING
0006 FOO PING
0007 FIZ PING
0009 FOO PING
0010 BAR PING
0011 BUZ PING
0012 FOO PING
0014 FIZ PING
0015 BAR PING
0015 FOO PING
0018 FOO PING
0020 BAR PING
0021 FIZ PING
0021 FOO PING
0022 BUZ PING
0024 FOO PING
0025 BAR PING
0027 FOO PING
0028 FIZ PING
0030 BAR PING
0030 FOO PING
0033 BUZ PING
0033 FOO PING
0035 FIZ PING
0035 BAR PING
0036 FOO PING
0039 FOO PING
0040 BAR PING
0042 FIZ PING
0042 FOO PING
0044 BUZ PING
0045 BAR PING
0045 FOO PING
0048 FOO PING
0049 FIZ PING
0050 BAR PING
0051 FOO PING
0054 FOO PING
0055 BUZ PING
0055 BAR PING
0056 FIZ PING
0057 FOO PING
0060 BAR PING
0060 FOO PING
0063 FIZ PING
0063 FOO PING
0065 BAR PING
0066 BUZ PING
0066 FOO PING
0069 FOO PING
0070 FIZ PING
0070 BAR PING
0072 FOO PING
0075 BAR PING
0075 FOO PING
0077 BUZ PING
0077 FIZ PING
0078 FOO PING
0080 BAR PING
0081 FOO PING
0084 FIZ PING
0084 FOO PING
0085 BAR PING
0087 FOO PING
0088 BUZ PING
0090 BAR PING
0090 FOO PING
0091 FIZ PING
0093 FOO PING
0095 BAR PING
0096 FOO PING
0098 FIZ PING
0099 BUZ PING
0099 FOO PING
0100 MAIN finish
"""


def test_finish(capsys):
    loop.reset()

    loop.add_initial(main())
    loop.add_initial(ping("FOO", 3))
    loop.add_initial(ping("BAR", 5))
    loop.add_initial(ping("FIZ", 7))
    loop.add_initial(ping("BUZ", 11))

    loop.run()

    # Subsequent calls to run() have no effect
    loop.run()

    assert capsys.readouterr().out == EXP1
