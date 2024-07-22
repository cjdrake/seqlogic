"""Test Ripple Carry Addition (RCA) algorithm."""

import os

from vcd import VCDWriter

from seqlogic import Module, Vec, active, cat, get_loop, simify, sleep, uint2vec
from seqlogic.algorithms.addition.rca import add

DIR = os.path.dirname(__file__)


def test_basic():
    for i in range(16):
        for j in range(16):
            for k in range(2):
                # Inputs
                a = uint2vec(i, 4)
                b = uint2vec(j, 4)
                ci = uint2vec(k, 1)

                # Outputs
                s, co = add(a, b, ci)

                # Check outputs
                q, r = divmod(i + j + k, 16)
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
        self.combi(s, lambda a, b, ci: a ^ b ^ ci, a, b, ci)
        self.combi(co, lambda a, b, ci: a & b | a & ci | b & ci, a, b, ci)


class RCA(Module):
    """Ripple Carry Adder."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        s = self.output(name="s", dtype=Vec[4])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[4])
        b = self.input(name="b", dtype=Vec[4])
        ci = self.input(name="ci", dtype=Vec[1])

        # Logic
        s0 = self.logic(name="s0", dtype=Vec[1])
        s1 = self.logic(name="s1", dtype=Vec[1])
        s2 = self.logic(name="s2", dtype=Vec[1])
        s3 = self.logic(name="s3", dtype=Vec[1])

        c0 = self.logic(name="c0", dtype=Vec[1])
        c1 = self.logic(name="c1", dtype=Vec[1])
        c2 = self.logic(name="c2", dtype=Vec[1])

        self.combi(s, cat, s0, s1, s2, s3)

        # Submodules
        self.submod(
            name="fa_0",
            mod=FullAdd,
        ).connect(
            s=s0,
            co=c0,
            a=(lambda x: x[0], a),
            b=(lambda x: x[0], b),
            ci=ci,
        )

        self.submod(
            name="fa_1",
            mod=FullAdd,
        ).connect(
            s=s1,
            co=c1,
            a=(lambda x: x[1], a),
            b=(lambda x: x[1], b),
            ci=c0,
        )

        self.submod(
            name="fa_2",
            mod=FullAdd,
        ).connect(
            s=s2,
            co=c2,
            a=(lambda x: x[2], a),
            b=(lambda x: x[2], b),
            ci=c1,
        )

        self.submod(
            name="fa_3",
            mod=FullAdd,
        ).connect(
            s=s3,
            co=co,
            a=(lambda x: x[3], a),
            b=(lambda x: x[3], b),
            ci=c2,
        )


class Top(Module):
    """Top Level Module."""

    def __init__(self, name: str):
        super().__init__(name, parent=None)

        s = self.output(name="s", dtype=Vec[4])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[4])
        b = self.input(name="b", dtype=Vec[4])
        ci = self.input(name="ci", dtype=Vec[1])

        self.submod(
            name="dut",
            mod=RCA,
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

        for i in range(16):
            for j in range(16):
                for k in range(2):
                    # Inputs
                    self.a.next = uint2vec(i, 4)
                    self.b.next = uint2vec(j, 4)
                    self.ci.next = uint2vec(k, 1)

                    await sleep(1)

                    # Check outputs
                    q, r = divmod(i + j + k, 16)
                    assert self.s.value.to_uint() == r
                    assert self.co.value.to_uint() == q

        await sleep(10)


def test_top():
    loop = get_loop()
    loop.reset()

    vcd = os.path.join(DIR, "rca.vcd")
    with open(vcd, "w", encoding="utf-8") as f:
        with VCDWriter(f, timescale="1ns") as vcdw:
            top = Top(name="top")
            top.dump_vcd(vcdw, ".*")
            simify(top)
            loop.run()
