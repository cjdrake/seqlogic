"""Data Path."""

# pyright: reportAttributeAccessIssue=false

import operator

from seqlogic import Module
from seqlogic.lbool import Vec, cat, rep, uint2vec

from . import TEXT_BASE, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst, Opcode
from .alu import Alu
from .regfile import RegFile


def f_pc_next(
    next_pc_sel: CtlPc, pc_plus_4: Vec[32], pc_plus_immediate: Vec[32], alu_result: Vec[32]
) -> Vec[32]:
    sel = next_pc_sel
    match sel:
        case CtlPc.PC4:
            return pc_plus_4
        case CtlPc.PC_IMM:
            return pc_plus_immediate
        case CtlPc.RS1_IMM:
            return cat("1b0", alu_result[1:])
        case _:
            return Vec[32].xprop(sel)


def f_alu_op_a(alu_op_a_sel: CtlAluA, rs1_data: Vec[32], pc: Vec[32]) -> Vec[32]:
    sel = alu_op_a_sel
    match sel:
        case CtlAluA.RS1:
            return rs1_data
        case CtlAluA.PC:
            return pc
        case _:
            return Vec[32].xprop(sel)


def f_alu_op_b(alu_op_b_sel: CtlAluB, rs2_data: Vec[32], immediate: Vec[32]) -> Vec[32]:
    sel = alu_op_b_sel
    match sel:
        case CtlAluB.RS2:
            return rs2_data
        case CtlAluB.IMM:
            return immediate
        case _:
            return Vec[32].xprop(sel)


def f_wr_data(
    reg_writeback_sel: CtlWriteBack,
    alu_result: Vec[32],
    data_mem_rd_data: Vec[32],
    pc_plus_4: Vec[32],
    immediate: Vec[32],
) -> Vec[32]:
    sel = reg_writeback_sel
    match sel:
        case CtlWriteBack.ALU:
            return alu_result
        case CtlWriteBack.DATA:
            return data_mem_rd_data
        case CtlWriteBack.PC4:
            return pc_plus_4
        case CtlWriteBack.IMM:
            return immediate
        case _:
            return Vec[32].xprop(sel)


def f_immediate(inst) -> Vec[32]:
    sel = inst.opcode
    match sel:
        case Opcode.LOAD | Opcode.LOAD_FP | Opcode.OP_IMM | Opcode.JALR:
            return cat(
                inst[20:31],
                rep(inst[31], 21),
            )
        case Opcode.STORE_FP | Opcode.STORE:
            return cat(
                inst[7:12],
                inst[25:31],
                rep(inst[31], 21),
            )
        case Opcode.BRANCH:
            return cat(
                "1b0",
                inst[8:12],
                inst[25:31],
                inst[7],
                rep(inst[31], 20),
            )
        case Opcode.AUIPC | Opcode.LUI:
            return cat(
                "12h000",
                inst[12:31],
                inst[31],
            )
        case Opcode.JAL:
            return cat(
                "1b0",
                inst[21:31],
                inst[20],
                inst[12:20],
                rep(inst[31], 12),
            )
        case _:
            return Vec[32].xprop(sel)


class DataPath(Module):
    """Data Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        data_mem_addr = self.bits(name="data_mem_addr", dtype=Vec[32], port=True)
        data_mem_wr_data = self.bits(name="data_mem_wr_data", dtype=Vec[32], port=True)
        data_mem_rd_data = self.bits(name="data_mem_rd_data", dtype=Vec[32], port=True)

        inst = self.bits(name="inst", dtype=Inst, port=True)
        pc = self.bits(name="pc", dtype=Vec[32], port=True)

        # Control=>Datapath
        pc_wr_en = self.bit(name="pc_wr_en", port=True)
        regfile_wr_en = self.bit(name="regfile_wr_en", port=True)
        alu_op_a_sel = self.bits(name="alu_op_a_sel", dtype=CtlAluA, port=True)
        alu_op_b_sel = self.bits(name="alu_op_b_sel", dtype=CtlAluB, port=True)
        alu_func = self.bits(name="alu_func", dtype=AluOp, port=True)
        reg_writeback_sel = self.bits(name="reg_writeback_sel", dtype=CtlWriteBack, port=True)
        next_pc_sel = self.bits(name="next_pc_sel", dtype=CtlPc, port=True)
        # Datapath=>Control
        alu_result_eq_zero = self.bit(name="alu_result_eq_zero", port=True)

        clock = self.bit(name="clock", port=True)
        reset = self.bit(name="reset", port=True)

        # Immedate generate
        immediate = self.bits(name="immediate", dtype=Vec[32])

        # PC + 4
        pc_plus_4 = self.bits(name="pc_plus_4", dtype=Vec[32])

        # PC + Immediate
        pc_plus_immediate = self.bits(name="pc_plus_immediate", dtype=Vec[32])

        # Select ALU Ops
        alu_op_a = self.bits(name="alu_op_a", dtype=Vec[32])
        alu_op_b = self.bits(name="alu_op_b", dtype=Vec[32])

        # ALU Outputs
        alu_result = self.bits(name="alu_result", dtype=Vec[32])

        # Next PC
        pc_next = self.bits(name="pc_next", dtype=Vec[32])

        # Regfile Write Data
        wr_data = self.bits(name="wr_data", dtype=Vec[32])

        # State
        rs1_data = self.bits(name="rs1_data", dtype=Vec[32])
        rs2_data = self.bits(name="rs2_data", dtype=Vec[32])

        # Submodules
        alu = self.submod(name="alu", mod=Alu)
        self.assign(alu_result, alu.result)
        self.assign(alu_result_eq_zero, alu.result_eq_zero)
        self.assign(alu.alu_func, alu_func)
        self.assign(alu.op_a, alu_op_a)
        self.assign(alu.op_b, alu_op_b)

        regfile = self.submod(name="regfile", mod=RegFile)
        self.assign(regfile.wr_en, regfile_wr_en)
        self.assign(regfile.wr_data, wr_data)
        self.assign(rs1_data, regfile.rs1_data)
        self.assign(rs2_data, regfile.rs2_data)
        self.assign(regfile.clock, clock)

        self.assign(data_mem_addr, alu_result)
        self.assign(data_mem_wr_data, rs2_data)

        # Combinational Logic
        ys = (regfile.rs2_addr, regfile.rs1_addr, regfile.wr_addr)
        self.combi(ys, lambda x: (x.rs2, x.rs1, x.rd), inst)
        self.combi(pc_plus_4, lambda x: x + "32h0000_0004", pc)
        self.combi(pc_plus_immediate, operator.add, pc, immediate)
        self.combi(pc_next, f_pc_next, next_pc_sel, pc_plus_4, pc_plus_immediate, alu_result)
        self.combi(alu_op_a, f_alu_op_a, alu_op_a_sel, rs1_data, pc)
        self.combi(alu_op_b, f_alu_op_b, alu_op_b_sel, rs2_data, immediate)
        xs = (reg_writeback_sel, alu_result, data_mem_rd_data, pc_plus_4, immediate)
        self.combi(wr_data, f_wr_data, *xs)
        self.combi(immediate, f_immediate, inst)

        # Sequential Logic
        self.dff_en_ar(pc, pc_next, pc_wr_en, clock, reset, uint2vec(TEXT_BASE, 32))
