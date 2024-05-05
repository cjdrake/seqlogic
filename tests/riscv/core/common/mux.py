"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed, clog2
from seqlogic.lbool import Vec
from seqlogic.sim import always_comb


class Mux(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, n: int, dtype: type[Vec]):
        super().__init__(name, parent)

        self._n = n
        self._dtype = dtype

        self.build()

    def build(self):
        # Ports
        self.out = Bits(name="out", parent=self, dtype=self._dtype)
        self.sel = Bits(name="sel", parent=self, dtype=Vec[clog2(self._n)])
        self.ins = [Bits(name=f"in{i}", parent=self, dtype=self._dtype) for i in range(self._n)]

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.sel, *self.ins)
            try:
                index = self.sel.value.to_uint()
            except ValueError:
                self.out.next = self._dtype.xes()
            else:
                self.out.next = self.ins[index].value
