"""Ripple Carry Addition (RCA)."""

from bvwx import Vec, cat

from seqlogic import GetItem, Module

from .fa import FullAdd


def adc(a: Vec, b: Vec, ci: Vec[1]) -> Vec:
    """Ripple Carry Addition."""
    n = len(a)
    assert n > 0 and n == len(b)

    # Carries
    c = [ci]
    for i, (a_i, b_i) in enumerate(zip(a, b)):
        c.append(a_i & b_i | c[i] & (a_i | b_i))
    c, co = cat(*c[:n]), c[n]

    # Sum
    s = a ^ b ^ c

    return cat(s, co)


class RCA(Module):
    """Ripple Carry Adder."""

    N: int = 8

    def build(self):
        # Ports
        s = self.output(name="s", dtype=Vec[self.N])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[self.N])
        b = self.input(name="b", dtype=Vec[self.N])
        ci = self.input(name="ci", dtype=Vec[1])

        # Unpacked sum bits
        ss = [self.logic(name=f"s{i}", dtype=Vec[1]) for i in range(self.N)]

        # Carries
        cs = [self.logic(name=f"c{i}", dtype=Vec[1]) for i in range(self.N)]

        # Pack sum bits
        self.combi(s, cat, *ss)

        # Instantiate and connect N FullAdd submodules
        for i in range(self.N):
            self.submod(
                name=f"fa_{i}",
                mod=FullAdd,
            ).connect(
                s=ss[i],
                co=(co if i == (self.N - 1) else cs[i]),
                a=GetItem(a, i),
                b=GetItem(b, i),
                ci=(ci if i == 0 else cs[i - 1]),
            )
