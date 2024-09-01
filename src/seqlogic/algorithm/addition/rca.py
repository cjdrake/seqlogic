"""Ripple Carry Addition (RCA)."""

from ...bits import Vector as Vec
from ...bits import cat
from ...design import Module
from .fa import FullAdd


def add(a: Vec, b: Vec, ci: Vec[1]) -> tuple[Vec, Vec[1]]:
    """Ripple Carry Addition."""
    n = len(a)
    assert n > 0 and n == len(b)

    # Carries
    c = [ci]
    for i, (a_i, b_i) in enumerate(zip(a, b)):
        c.append(a_i & b_i | c[i] & (a_i | b_i))
    c, co = cat(*c[:n]), c[n]

    return a ^ b ^ c, co


class RCA(Module):
    """Ripple Carry Adder."""

    n: int = 8

    def build(self):
        n = self.n

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
