"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.logicvec import F, T, xes, zeros
from seqlogic.sim import notify
from seqlogic.var import Bit, LogicVec

from ..misc import COMBI
from .constants import AluOp


class Alu(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.result = LogicVec(name="result", parent=self, shape=(32,))
        self.result_equal_zero = Bit(name="result_equal_zero", parent=self)
        self.alu_function = LogicVec(name="alu_function", parent=self, shape=(5,))
        self.op_a = LogicVec(name="op_a", parent=self, shape=(32,))
        self.op_b = LogicVec(name="op_b", parent=self, shape=(32,))

        # Processes
        self._procs.add((self.proc_result, COMBI))
        self._procs.add((self.proc_result_equal_zero, COMBI))

    async def proc_result(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.alu_function.changed, self.op_a.changed, self.op_b.changed)
            match self.alu_function.next:
                case AluOp.ADD:
                    self.result.next = self.op_a.next + self.op_b.next
                case AluOp.SUB:
                    self.result.next = self.op_a.next - self.op_b.next
                case AluOp.SLL:
                    self.result.next = self.op_a.next << self.op_b.next[0:5]
                case AluOp.SRL:
                    self.result.next = self.op_a.next >> self.op_b.next[0:5]
                case AluOp.SRA:
                    self.result.next, _ = self.op_a.next.arsh(self.op_b.next[0:5])
                case AluOp.SEQ:
                    try:
                        a = self.op_a.next.to_uint()
                        b = self.op_b.next.to_uint()
                    except ValueError:
                        self.result.next = xes((32,))
                    else:
                        x = T if a == b else F
                        self.result.next = x.zext(31)
                case AluOp.SLT:
                    try:
                        a = self.op_a.next.to_int()
                        b = self.op_b.next.to_int()
                    except ValueError:
                        self.result.next = xes((32,))
                    else:
                        x = T if a < b else F
                        self.result.next = x.zext(31)
                case AluOp.SLTU:
                    try:
                        a = self.op_a.next.to_uint()
                        b = self.op_b.next.to_uint()
                    except ValueError:
                        self.result.next = xes((32,))
                    else:
                        x = T if a < b else F
                        self.result.next = x.zext(31)
                case AluOp.XOR:
                    self.result.next = self.op_a.next ^ self.op_b.next
                case AluOp.OR:
                    self.result.next = self.op_a.next | self.op_b.next
                case AluOp.AND:
                    self.result.next = self.op_a.next & self.op_b.next
                case _:
                    self.result.next = zeros((32,))

    async def proc_result_equal_zero(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.result.changed)
            if self.result.next == zeros((32,)):
                self.result_equal_zero.next = T
            else:
                self.result_equal_zero.next = F
