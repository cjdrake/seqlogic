"""Arithmetic Logic Unit (ALU)."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Bit, Bits, Module
from seqlogic.lbool import Vec, dcs, xes

from . import AluOp


def alu_result(op: AluOp, a: Vec[32], b: Vec[32]) -> Vec[32]:
    match op:
        case AluOp.ADD:
            return a + b
        case AluOp.SUB:
            return a - b
        case AluOp.SLL:
            return a << b[0:5]
        case AluOp.SRL:
            return a >> b[0:5]
        case AluOp.SRA:
            y, _ = a.arsh(b[0:5])
            return y
        case AluOp.SEQ:
            return a.eq(b).zext(32 - 1)
        case AluOp.SLT:
            return a.lt(b).zext(32 - 1)
        case AluOp.SLTU:
            return a.ltu(b).zext(32 - 1)
        case AluOp.XOR:
            return a ^ b
        case AluOp.OR:
            return a | b
        case AluOp.AND:
            return a & b
        case _:
            return xes(32) if op.has_x() else dcs(32)


class Alu(Module):
    """Arithmetic Logic Unit (ALU)."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        result = Bits(name="result", parent=self, dtype=Vec[32])
        result_eq_zero = Bit(name="result_eq_zero", parent=self)
        alu_func = Bits(name="alu_func", parent=self, dtype=AluOp)
        op_a = Bits(name="op_a", parent=self, dtype=Vec[32])
        op_b = Bits(name="op_b", parent=self, dtype=Vec[32])

        self.combi(result, alu_result, alu_func, op_a, op_b)
        self.combi(result_eq_zero, lambda x: x.eq("32h0000_0000"), result)

        # TODO(cjdrake): Remove
        self.result = result
        self.result_eq_zero = result_eq_zero
        self.alu_func = alu_func
        self.op_a = op_a
        self.op_b = op_b
