"""Data Path."""

from seqlogic import Bit, Bits, Module, changed, resume
from seqlogic.lbool import Vec, cat, rep, uint2vec
from seqlogic.sim import active, reactive

from . import TEXT_BASE, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst, Opcode
from .alu import Alu
from .regfile import RegFile


class DataPath(Module):
    """Data Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._build()
        self._connect()

    def _build(self):
        self.data_mem_addr = Bits(name="data_mem_addr", parent=self, dtype=Vec[32])
        self.data_mem_wr_data = Bits(name="data_mem_wr_data", parent=self, dtype=Vec[32])
        self.data_mem_rd_data = Bits(name="data_mem_rd_data", parent=self, dtype=Vec[32])

        self.inst = Bits(name="inst", parent=self, dtype=Inst)
        self.pc = Bits(name="pc", parent=self, dtype=Vec[32])

        # Immedate generate
        self.immediate = Bits(name="immediate", parent=self, dtype=Vec[32])

        # PC + 4
        self.pc_plus_4 = Bits(name="pc_plus_4", parent=self, dtype=Vec[32])

        # PC + Immediate
        self.pc_plus_immediate = Bits(name="pc_plus_immediate", parent=self, dtype=Vec[32])

        # Control signals
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        self.alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        self.alu_function = Bits(name="alu_function", parent=self, dtype=AluOp)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)

        # Select ALU Ops
        self.alu_op_a = Bits(name="alu_op_a", parent=self, dtype=Vec[32])
        self.alu_op_b = Bits(name="alu_op_b", parent=self, dtype=Vec[32])

        # ALU Outputs
        self.alu_result = Bits(name="alu_result", parent=self, dtype=Vec[32])
        self.alu_result_eq_zero = Bit(name="alu_result_eq_zero", parent=self)

        # Next PC
        self.pc_next = Bits(name="pc_next", parent=self, dtype=Vec[32])

        # Regfile Write Data
        self.wr_data = Bits(name="wr_data", parent=self, dtype=Vec[32])

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self.rs1_data = Bits(name="rs1_data", parent=self, dtype=Vec[32])
        self.rs2_data = Bits(name="rs2_data", parent=self, dtype=Vec[32])

        # Submodules
        self.alu = Alu(name="alu", parent=self)
        self.regfile = RegFile(name="regfile", parent=self)

    def _connect(self):
        self.data_mem_addr.connect(self.alu_result)
        self.data_mem_wr_data.connect(self.rs2_data)

        self.alu_result.connect(self.alu.result)
        self.alu_result_eq_zero.connect(self.alu.result_eq_zero)
        self.alu.alu_function.connect(self.alu_function)
        self.alu.op_a.connect(self.alu_op_a)
        self.alu.op_b.connect(self.alu_op_b)

        self.regfile.wr_en.connect(self.regfile_wr_en)
        self.regfile.wr_data.connect(self.wr_data)
        self.rs1_data.connect(self.regfile.rs1_data)
        self.rs2_data.connect(self.regfile.rs2_data)
        self.regfile.clock.connect(self.clock)
        self.regfile.reset.connect(self.reset)

    @active
    async def p_f_0(self):
        def f():
            return self.clock.is_posedge() and self.reset.is_neg() and self.pc_wr_en.value == "1b1"

        while True:
            state = await resume((self.reset, self.reset.is_posedge), (self.clock, f))
            if state is self.reset:
                self.pc.next = uint2vec(TEXT_BASE, 32)
            elif state is self.clock:
                self.pc.next = self.pc_next.value
            else:
                assert False

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
