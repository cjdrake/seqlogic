"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed, clog2
from seqlogic.lbool import xes
from seqlogic.sim import always_comb


class Mux(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, n: int, shape: tuple[int, ...]):
        super().__init__(name, parent)

        self._n = n
        self._shape = shape

        self.build()

    def build(self):
        # Ports
        self.out = Bits(name="out", parent=self, shape=self._shape)
        self.sel = Bits(name="sel", parent=self, shape=(clog2(self._n),))
        self.ins = [Bits(name=f"in{i}", parent=self, shape=self._shape) for i in range(self._n)]

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.sel, *self.ins)
            try:
                index = self.sel.value.to_uint()
            except ValueError:
                self.out.next = xes(self._shape[0])
            else:
                self.out.next = self.ins[index].value
