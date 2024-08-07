"""Test Ripple Carry Addition (RCA) algorithm."""

import os

from vcd import VCDWriter

from seqlogic import Module, Vec, active, cat, get_loop, sleep, u2bv
from seqlogic.algorithms.addition.rca import add

DIR = os.path.dirname(__file__)

N_START = 1
N_STOP = 6


def test_functional():
    for n in range(N_START, N_STOP):
        for i in range(2**n):
            for j in range(2**n):
                for k in range(2**1):
                    # Inputs
                    a = u2bv(i, n)
                    b = u2bv(j, n)
                    ci = u2bv(k, 1)

                    # Outputs
                    s, co = add(a, b, ci)

                    # Check outputs
                    q, r = divmod(i + j + k, 2**n)
                    assert s.to_uint() == r
                    assert co.to_uint() == q


class FullAdd(Module):
    """Full Adder."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        s = self.output(name="s", dtype=Vec[1])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[1])
        b = self.input(name="b", dtype=Vec[1])
        ci = self.input(name="ci", dtype=Vec[1])

        # Combinational Logic
        self.expr(s, a ^ b ^ ci)
        self.expr(co, a & b | ci & (a | b))


class RCA(Module):
    """Ripple Carry Adder."""

    def __init__(self, name: str, parent: Module | None, n: int):
        assert n > 0

        super().__init__(name, parent)

        # Ports
        s = self.output(name="s", dtype=Vec[n])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[n])
        b = self.input(name="b", dtype=Vec[n])
        ci = self.input(name="ci", dtype=Vec[1])

        # Unpacked sum bits
        ss = [self.logic(name=f"s{i}", dtype=Vec[1]) for i in range(n)]

        # Carries
        cs = [self.logic(name=f"c{i}", dtype=Vec[1]) for i in range(n)]

        # Pack sum bits
        self.combi(s, cat, *ss)

        # Instantiate and connect N FullAdd submodules
        for i in range(n):
            self.submod(
                name=f"fa_{i}",
                mod=FullAdd,
            ).connect(
                s=ss[i],
                co=(co if i == (n - 1) else cs[i]),
                a=a[i],
                b=b[i],
                ci=(ci if i == 0 else cs[i - 1]),
            )


class Top(Module):
    """Top Level Module."""

    def __init__(self, name: str, n: int):
        super().__init__(name, parent=None)

        self.n = n

        s = self.output(name="s", dtype=Vec[n])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[n])
        b = self.input(name="b", dtype=Vec[n])
        ci = self.input(name="ci", dtype=Vec[1])

        # Design Under Test
        self.submod(
            name="dut",
            mod=RCA,
            n=n,
        ).connect(
            s=s,
            ci=ci,
            a=a,
            b=b,
            co=co,
        )

    @active
    async def drive(self):
        await sleep(10)

        n = self.n

        for i in range(2**n):
            for j in range(2**n):
                for k in range(2**1):
                    # Inputs
                    self.a.next = u2bv(i, n)
                    self.b.next = u2bv(j, n)
                    self.ci.next = u2bv(k, 1)

                    await sleep(1)

                    # Check outputs
                    q, r = divmod(i + j + k, 2**n)
                    assert self.s.value.to_uint() == r
                    assert self.co.value.to_uint() == q

        await sleep(10)


def test_structural():
    loop = get_loop()

    for n in range(N_START, N_STOP):
        loop.reset()

        vcd = os.path.join(DIR, "rca.vcd")
        with (
            open(vcd, "w", encoding="utf-8") as f,
            VCDWriter(f, timescale="1ns") as vcdw,
        ):
            # Instantiate top
            top = Top(name="top", n=n)

            # Dump all signals to VCD
            top.dump_vcd(vcdw, ".*")

            # Register design w/ event loop
            top.elab()

            # Do the damn thing
            loop.run()
