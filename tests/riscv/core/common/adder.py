"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, notify

from ..misc import COMBI


class Adder(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Parameters
        assert width > 1
        self.width = width

        self.build()

        # Processes
        self._procs.add((self.proc_out, COMBI))

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.result = Bits(name="result", parent=self, shape=(self.width,))
        self.op_a = Bits(name="op_a", parent=self, shape=(self.width,))
        self.op_b = Bits(name="op_b", parent=self, shape=(self.width,))

    async def proc_out(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.op_a.changed, self.op_b.changed)
            self.result.next = self.op_a.next + self.op_b.next
