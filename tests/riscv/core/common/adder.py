"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, notify
from seqlogic.sim import always_comb


class Adder(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Parameters
        assert width > 1
        self.width = width

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        self.result = Bits(name="result", parent=self, shape=(self.width,))
        self.op_a = Bits(name="op_a", parent=self, shape=(self.width,))
        self.op_b = Bits(name="op_b", parent=self, shape=(self.width,))

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.op_a.changed, self.op_b.changed)
            self.result.next = self.op_a.next + self.op_b.next
