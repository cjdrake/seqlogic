"""Data Path."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, cat, rep, uint2vec
from seqlogic.sim import reactive

from . import TEXT_BASE, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst, Opcode
from .alu import Alu
from .regfile import RegFile


class DataPath(Module):
    """Data Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        data_mem_addr = Bits(name="data_mem_addr", parent=self, dtype=Vec[32])
        data_mem_wr_data = Bits(name="data_mem_wr_data", parent=self, dtype=Vec[32])
        data_mem_rd_data = Bits(name="data_mem_rd_data", parent=self, dtype=Vec[32])

        inst = Bits(name="inst", parent=self, dtype=Inst)
        pc = Bits(name="pc", parent=self, dtype=Vec[32])

        # Immedate generate
        immediate = Bits(name="immediate", parent=self, dtype=Vec[32])

        # PC + 4
        pc_plus_4 = Bits(name="pc_plus_4", parent=self, dtype=Vec[32])

        # PC + Immediate
        pc_plus_immediate = Bits(name="pc_plus_immediate", parent=self, dtype=Vec[32])

        # Control signals
        pc_wr_en = Bit(name="pc_wr_en", parent=self)
        regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        alu_func = Bits(name="alu_func", parent=self, dtype=AluOp)
        reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)

        # Select ALU Ops
        alu_op_a = Bits(name="alu_op_a", parent=self, dtype=Vec[32])
        alu_op_b = Bits(name="alu_op_b", parent=self, dtype=Vec[32])

        # ALU Outputs
        alu_result = Bits(name="alu_result", parent=self, dtype=Vec[32])
        alu_result_eq_zero = Bit(name="alu_result_eq_zero", parent=self)

        # Next PC
        pc_next = Bits(name="pc_next", parent=self, dtype=Vec[32])

        # Regfile Write Data
        wr_data = Bits(name="wr_data", parent=self, dtype=Vec[32])

        clock = Bit(name="clock", parent=self)
        reset = Bit(name="reset", parent=self)

        # State
        rs1_data = Bits(name="rs1_data", parent=self, dtype=Vec[32])
        rs2_data = Bits(name="rs2_data", parent=self, dtype=Vec[32])

        # Submodules
        alu = Alu(name="alu", parent=self)
        self.connect(alu_result, alu.result)
        self.connect(alu_result_eq_zero, alu.result_eq_zero)
        self.connect(alu.alu_func, alu_func)
        self.connect(alu.op_a, alu_op_a)
        self.connect(alu.op_b, alu_op_b)

        regfile = RegFile(name="regfile", parent=self)
        self.connect(regfile.wr_en, regfile_wr_en)
        self.connect(regfile.wr_data, wr_data)
        self.connect(rs1_data, regfile.rs1_data)
        self.connect(rs2_data, regfile.rs2_data)
        self.connect(regfile.clock, clock)
        self.connect(regfile.reset, reset)

        self.connect(data_mem_addr, alu_result)
        self.connect(data_mem_wr_data, rs2_data)

        self.dff_en_ar(pc, pc_next, pc_wr_en, clock, reset, uint2vec(TEXT_BASE, 32))

        # TODO(cjdrake): Remove
        self.data_mem_addr = data_mem_addr
        self.data_mem_wr_data = data_mem_wr_data
        self.data_mem_rd_data = data_mem_rd_data
        self.inst = inst
        self.pc = pc
        self.immediate = immediate
        self.pc_plus_4 = pc_plus_4
        self.pc_plus_immediate = pc_plus_immediate
        self.pc_wr_en = pc_wr_en
        self.regfile_wr_en = regfile_wr_en
        self.alu_op_a_sel = alu_op_a_sel
        self.alu_op_b_sel = alu_op_b_sel
        self.alu_func = alu_func
        self.reg_writeback_sel = reg_writeback_sel
        self.next_pc_sel = next_pc_sel
        self.alu_op_a = alu_op_a
        self.alu_op_b = alu_op_b
        self.alu_result = alu_result
        self.alu_result_eq_zero = alu_result_eq_zero
        self.pc_next = pc_next
        self.wr_data = wr_data
        self.clock = clock
        self.reset = reset
        self.rs1_data = rs1_data
        self.rs2_data = rs2_data
        self.regfile = regfile

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst)
            self.regfile.rs2_addr.next = self.inst.value.rs2
            self.regfile.rs1_addr.next = self.inst.value.rs1
            self.regfile.wr_addr.next = self.inst.value.rd

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.pc)
            self.pc_plus_4.next = self.pc.value + "32h0000_0004"

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.pc, self.immediate)
            self.pc_plus_immediate.next = self.pc.value + self.immediate.value

    @reactive
    async def p_c_3(self):
        while True:
            await changed(self.next_pc_sel, self.pc_plus_4, self.pc_plus_immediate, self.alu_result)
            sel = self.next_pc_sel.value
            match sel:
                case CtlPc.PC4:
                    self.pc_next.next = self.pc_plus_4.value
                case CtlPc.PC_IMM:
                    self.pc_next.next = self.pc_plus_immediate.value
                case CtlPc.RS1_IMM:
                    self.pc_next.next = cat("1b0", self.alu_result.value[1:])
                case _:
                    self.pc_next.xprop(sel)

    @reactive
    async def p_c_4(self):
        while True:
            await changed(self.alu_op_a_sel, self.rs1_data, self.pc)
            sel = self.alu_op_a_sel.value
            match sel:
                case CtlAluA.RS1:
                    self.alu_op_a.next = self.rs1_data.value
                case CtlAluA.PC:
                    self.alu_op_a.next = self.pc.value
                case _:
                    self.alu_op_a.xprop(sel)

    @reactive
    async def p_c_5(self):
        while True:
            await changed(self.alu_op_b_sel, self.rs2_data, self.immediate)
            sel = self.alu_op_b_sel.value
            match sel:
                case CtlAluB.RS2:
                    self.alu_op_b.next = self.rs2_data.value
                case CtlAluB.IMM:
                    self.alu_op_b.next = self.immediate.value
                case _:
                    self.alu_op_b.xprop(sel)

    @reactive
    async def p_c_6(self):
        while True:
            await changed(
                self.reg_writeback_sel,
                self.alu_result,
                self.data_mem_rd_data,
                self.pc_plus_4,
                self.immediate,
            )
            sel = self.reg_writeback_sel.value
            match sel:
                case CtlWriteBack.ALU:
                    self.wr_data.next = self.alu_result.value
                case CtlWriteBack.DATA:
                    self.wr_data.next = self.data_mem_rd_data.value
                case CtlWriteBack.PC4:
                    self.wr_data.next = self.pc_plus_4.value
                case CtlWriteBack.IMM:
                    self.wr_data.next = self.immediate.value
                case _:
                    self.wr_data.xprop(sel)

    @reactive
    async def p_c_7(self):
        while True:
            await changed(self.inst)
            sel = self.inst.value.opcode
            match sel:
                case Opcode.LOAD | Opcode.LOAD_FP | Opcode.OP_IMM | Opcode.JALR:
                    self.immediate.next = cat(
                        self.inst.value[20:31],
                        rep(self.inst.value[31], 21),
                    )
                case Opcode.STORE_FP | Opcode.STORE:
                    self.immediate.next = cat(
                        self.inst.value[7:12],
                        self.inst.value[25:31],
                        rep(self.inst.value[31], 21),
                    )
                case Opcode.BRANCH:
                    self.immediate.next = cat(
                        "1b0",
                        self.inst.value[8:12],
                        self.inst.value[25:31],
                        self.inst.value[7],
                        rep(self.inst.value[31], 20),
                    )
                case Opcode.AUIPC | Opcode.LUI:
                    self.immediate.next = cat(
                        "12h000",
                        self.inst.value[12:31],
                        self.inst.value[31],
                    )
                case Opcode.JAL:
                    self.immediate.next = cat(
                        "1b0",
                        self.inst.value[21:31],
                        self.inst.value[20],
                        self.inst.value[12:20],
                        rep(self.inst.value[31], 12),
                    )
                case _:
                    self.immediate.xprop(sel)
