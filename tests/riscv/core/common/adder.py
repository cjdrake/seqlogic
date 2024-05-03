"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed
from seqlogic.sim import always_comb


class Adder(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        super().__init__(name, parent)

        # Parameters
        assert width > 1
        self._width = width

        self.build()

    def build(self):
        self.result = Bits(name="result", parent=self, shape=(self._width,))
        self.op_a = Bits(name="op_a", parent=self, shape=(self._width,))
        self.op_b = Bits(name="op_b", parent=self, shape=(self._width,))

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.op_a, self.op_b)
            self.result.next = self.op_a.value + self.op_b.value
