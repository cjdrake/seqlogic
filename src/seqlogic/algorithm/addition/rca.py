"""Ripple Carry Addition (RCA)."""

from ...bits import Vector as Vec
from ...bits import cat
from ...design import Module
from .fa import FullAdd


def add(a: Vec, b: Vec, ci: Vec[1]) -> tuple[Vec, Vec[1]]:
    """Ripple Carry Addition."""
    assert len(a) > 0 and len(a) == len(b)

    gen = zip(a, b)

    a_0, b_0 = next(gen)
    s = [a_0 ^ b_0 ^ ci]
    c = [a_0 & b_0 | ci & (a_0 | b_0)]

    for i, (a_i, b_i) in enumerate(gen, start=1):
        s.append(a_i ^ b_i ^ c[i - 1])
        c.append(a_i & b_i | c[i - 1] & (a_i | b_i))

    return cat(*s), c[-1]


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
