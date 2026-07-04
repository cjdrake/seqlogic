"""Full Adder Module."""

from bvwx import Array

from seqlogic import Module


class FullAdd(Module):
    """Full Adder."""

    def build(self):
        # Ports
        s = self.output(name="s", dtype=Array[1])
        co = self.output(name="co", dtype=Array[1])

        a = self.input(name="a", dtype=Array[1])
        b = self.input(name="b", dtype=Array[1])
        ci = self.input(name="ci", dtype=Array[1])

        # Combinational Logic
        self.expr(s, a ^ b ^ ci)
        self.expr(co, a & b | ci & (a | b))
