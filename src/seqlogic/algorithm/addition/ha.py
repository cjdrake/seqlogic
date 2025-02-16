"""Half Adder Module."""

from bvwx import Vec

from seqlogic import Module


class HalfAdd(Module):
    """Half Adder."""

    def build(self):
        # Ports
        s = self.output(name="s", dtype=Vec[1])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[1])
        b = self.input(name="b", dtype=Vec[1])

        # Combinational Logic
        self.expr(s, a ^ b)
        self.expr(co, a & b)
