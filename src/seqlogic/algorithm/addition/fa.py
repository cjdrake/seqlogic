"""Full Adder Module."""

from seqlogic import Module, Vec


class FullAdd(Module):
    """Full Adder."""

    def build(self):
        # Ports
        s = self.output(name="s", dtype=Vec[1])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[1])
        b = self.input(name="b", dtype=Vec[1])
        ci = self.input(name="ci", dtype=Vec[1])

        # Combinational Logic
        self.expr(s, a ^ b ^ ci)
        self.expr(co, a & b | ci & (a | b))
