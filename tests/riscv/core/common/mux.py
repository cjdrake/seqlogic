"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, notify
from seqlogic.bits import xes
from seqlogic.sim import always_comb
from seqlogic.util import clog2


class Mux(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, n: int, width: int):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.n = n
        self.width = width

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.out = Bits(name="out", parent=self, shape=(self.width,))
        self.sel = Bits(name="sel", parent=self, shape=(clog2(self.n),))
        self.ins = [Bits(name=f"in{i}", parent=self, shape=(self.width,)) for i in range(self.n)]

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        ins_changed = [x.changed for x in self.ins]
        while True:
            await notify(self.sel.changed, *ins_changed)
            try:
                index = self.sel.next.to_uint()
            except ValueError:
                self.out.next = xes((self.width,))
            else:
                self.out.next = self.ins[index].next
