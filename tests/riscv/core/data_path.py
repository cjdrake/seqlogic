"""Data Path."""

from seqlogic import Add, Equal, GetAttr, Module, Vec, cat, rep, u2bv

from . import TEXT_BASE, Addr, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst, Opcode
from .alu import Alu
from .regfile import RegFile


def f_pc_next(
    next_pc_sel: CtlPc, pc_plus_4: Vec[32], pc_plus_immediate: Vec[32], alu_result: Vec[32]
) -> Vec[32]:
    match next_pc_sel:
        case CtlPc.PC4:
            return pc_plus_4
        case CtlPc.PC_IMM:
            return pc_plus_immediate
        case CtlPc.RS1_IMM:
            return cat("1b0", alu_result[1:])
        case _:
            return Vec[32].xprop(next_pc_sel)


def f_alu_op_a(alu_op_a_sel: CtlAluA, rs1_data: Vec[32], pc: Vec[32]) -> Vec[32]:
    match alu_op_a_sel:
        case CtlAluA.RS1:
            return rs1_data
        case CtlAluA.PC:
            return pc
        case _:
            return Vec[32].xprop(alu_op_a_sel)


def f_alu_op_b(alu_op_b_sel: CtlAluB, rs2_data: Vec[32], immediate: Vec[32]) -> Vec[32]:
    match alu_op_b_sel:
        case CtlAluB.RS2:
            return rs2_data
        case CtlAluB.IMM:
            return immediate
        case _:
            return Vec[32].xprop(alu_op_b_sel)


def f_wr_data(
    reg_wr_sel: CtlWriteBack,
    alu_result: Vec[32],
    data_mem_rd_data: Vec[32],
    pc_plus_4: Vec[32],
    immediate: Vec[32],
) -> Vec[32]:
    match reg_wr_sel:
        case CtlWriteBack.ALU:
            return alu_result
        case CtlWriteBack.DATA:
            return data_mem_rd_data
        case CtlWriteBack.PC4:
            return pc_plus_4
        case CtlWriteBack.IMM:
            return immediate
        case _:
            return Vec[32].xprop(reg_wr_sel)


def f_immediate(inst) -> Vec[32]:
    match inst.opcode:
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
            return Vec[32].xprop(inst.opcode)


class DataPath(Module):
    """Data Path Module."""

    def build(self):
        data_mem_addr = self.output(name="data_mem_addr", dtype=Addr)
        data_mem_wr_data = self.output(name="data_mem_wr_data", dtype=Vec[32])
        data_mem_rd_data = self.input(name="data_mem_rd_data", dtype=Vec[32])

        inst = self.input(name="inst", dtype=Inst)
        pc = self.output(name="pc", dtype=Vec[32])

        # Datapath=>Control
        alu_result_eq_zero = self.output(name="alu_result_eq_zero", dtype=Vec[1])

        # Control=>Datapath
        pc_wr_en = self.input(name="pc_wr_en", dtype=Vec[1])
        reg_wr_en = self.input(name="reg_wr_en", dtype=Vec[1])
        alu_op_a_sel = self.input(name="alu_op_a_sel", dtype=CtlAluA)
        alu_op_b_sel = self.input(name="alu_op_b_sel", dtype=CtlAluB)
        alu_op = self.input(name="alu_op", dtype=AluOp)
        reg_wr_sel = self.input(name="reg_wr_sel", dtype=CtlWriteBack)
        next_pc_sel = self.input(name="next_pc_sel", dtype=CtlPc)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        # Immedate generate
        immediate = self.logic(name="immediate", dtype=Vec[32])

        # PC + 4
        pc_plus_4 = self.logic(name="pc_plus_4", dtype=Vec[32])

        # PC + Immediate
        pc_plus_immediate = self.logic(name="pc_plus_immediate", dtype=Vec[32])

        # Select ALU Ops
        alu_op_a = self.logic(name="alu_op_a", dtype=Vec[32])
        alu_op_b = self.logic(name="alu_op_b", dtype=Vec[32])

        # ALU Outputs
        alu_result = self.logic(name="alu_result", dtype=Vec[32])

        # Next PC
        pc_next = self.logic(name="pc_next", dtype=Vec[32])

        # Regfile Write
        wr_data = self.logic(name="wr_data", dtype=Vec[32])

        # Regfile Read
        rs1_data = self.logic(name="rs1_data", dtype=Vec[32])
        rs2_data = self.logic(name="rs2_data", dtype=Vec[32])

        # Submodules
        self.submod(
            name="alu",
            mod=Alu,
        ).connect(
            y=alu_result,
            op=alu_op,
            a=alu_op_a,
            b=alu_op_b,
        )

        self.submod(
            name="regfile",
            mod=RegFile,
        ).connect(
            wr_en=reg_wr_en,
            wr_addr=GetAttr(inst, "rd"),
            wr_data=wr_data,
            rs1_addr=GetAttr(inst, "rs1"),
            rs1_data=rs1_data,
            rs2_addr=GetAttr(inst, "rs2"),
            rs2_data=rs2_data,
            clock=clock,
        )

        # Combinational Logic
        self.expr(alu_result_eq_zero, Equal(alu_result, u2bv(0, 32)))

        self.assign(data_mem_addr, alu_result)
        self.assign(data_mem_wr_data, rs2_data)

        self.combi(immediate, f_immediate, inst)

        self.expr(pc_plus_4, Add(pc, u2bv(4, 32)))
        self.expr(pc_plus_immediate, Add(pc, immediate))
        self.combi(pc_next, f_pc_next, next_pc_sel, pc_plus_4, pc_plus_immediate, alu_result)

        self.combi(alu_op_a, f_alu_op_a, alu_op_a_sel, rs1_data, pc)
        self.combi(alu_op_b, f_alu_op_b, alu_op_b_sel, rs2_data, immediate)

        xs = (reg_wr_sel, alu_result, data_mem_rd_data, pc_plus_4, immediate)
        self.combi(wr_data, f_wr_data, *xs)

        # Sequential Logic
        pc_rval = u2bv(TEXT_BASE, 32)
        self.dff_en_r(pc, pc_next, pc_wr_en, clock, reset, pc_rval)
