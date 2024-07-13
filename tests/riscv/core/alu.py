"""Arithmetic Logic Unit (ALU)."""

from seqlogic import Module
from seqlogic import Vector as Vec
from seqlogic import add, sub

from . import AluOp


def f(op: AluOp, a: Vec[32], b: Vec[32]) -> Vec[32]:
    match op:
        case AluOp.ADD:
            return add(a, b).s
        case AluOp.SUB:
            return sub(a, b).s
        case AluOp.SLL:
            return a << b[0:5]
        case AluOp.SRL:
            return a >> b[0:5]
        case AluOp.SRA:
            y, _ = a.srsh(b[0:5])
            return y
        case AluOp.SEQ:
            return a.eq(b).xt(32 - 1)
        case AluOp.SLT:
            return a.slt(b).xt(32 - 1)
        case AluOp.SLTU:
            return a.lt(b).xt(32 - 1)
        case AluOp.XOR:
            return a ^ b
        case AluOp.OR:
            return a | b
        case AluOp.AND:
            return a & b
        case _:
            return Vec[32].xprop(op)


class Alu(Module):
    """Arithmetic Logic Unit (ALU)."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        y = self.output(name="y", dtype=Vec[32])
        op = self.input(name="op", dtype=AluOp)
        a = self.input(name="a", dtype=Vec[32])
        b = self.input(name="b", dtype=Vec[32])

        # Combinational Logic
        self.combi(y, f, op, a, b)
