"""RiscV Core."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive

from .. import WORD_BITS, WORD_BYTES, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst
from ..common.data_mem_if import DataMemIf
from .ctl_path import CtlPath
from .data_path import DataPath


class Core(Module):
    """RiscV Core."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._build()
        self._connect()

    def _build(self):
        # Ports
        self.bus_addr = Bits(name="bus_addr", parent=self, dtype=Vec[32])
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, dtype=Vec[WORD_BYTES])
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, dtype=Vec[WORD_BITS])
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, dtype=Vec[WORD_BITS])

        self.pc = Bits(name="pc", parent=self, dtype=Vec[32])
        self.inst = Bits(name="inst", parent=self, dtype=Inst)

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self._pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self._regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self._alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        self._alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        self._reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        self._next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)
        self._alu_function = Bits(name="alu_function", parent=self, dtype=AluOp)
        self._alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)
        self._addr = Bits(name="addr", parent=self, dtype=Vec[32])
        self._wr_en = Bit(name="wr_en", parent=self)
        self._wr_data = Bits(name="wr_data", parent=self, dtype=Vec[WORD_BITS])
        self._rd_en = Bit(name="rd_en", parent=self)
        self._rd_data = Bits(name="rd_data", parent=self, dtype=Vec[WORD_BITS])

        # Submodules
        self.ctlpath = CtlPath(name="ctlpath", parent=self)
        self.datapath = DataPath(name="datapath", parent=self)
        self.data_mem_if = DataMemIf(name="data_mem_if", parent=self)

    def _connect(self):
        self.ctlpath.alu_result_equal_zero.connect(self._alu_result_equal_zero)
        self._pc_wr_en.connect(self.ctlpath.pc_wr_en)
        self._regfile_wr_en.connect(self.ctlpath.regfile_wr_en)
        self._alu_op_a_sel.connect(self.ctlpath.alu_op_a_sel)
        self._alu_op_b_sel.connect(self.ctlpath.alu_op_b_sel)
        self._rd_en.connect(self.ctlpath.data_mem_rd_en)
        self._wr_en.connect(self.ctlpath.data_mem_wr_en)
        self._reg_writeback_sel.connect(self.ctlpath.reg_writeback_sel)
        self._alu_function.connect(self.ctlpath.alu_function)
        self._next_pc_sel.connect(self.ctlpath.next_pc_sel)

        self._addr.connect(self.datapath.data_mem_addr)
        self._wr_data.connect(self.datapath.data_mem_wr_data)
        self.datapath.data_mem_rd_data.connect(self._rd_data)
        self.datapath.inst.connect(self.inst)
        self.pc.connect(self.datapath.pc)
        self._alu_result_equal_zero.connect(self.datapath.alu_result_equal_zero)
        self.datapath.pc_wr_en.connect(self._pc_wr_en)
        self.datapath.regfile_wr_en.connect(self._regfile_wr_en)
        self.datapath.alu_op_a_sel.connect(self._alu_op_a_sel)
        self.datapath.alu_op_b_sel.connect(self._alu_op_b_sel)
        self.datapath.reg_writeback_sel.connect(self._reg_writeback_sel)
        self.datapath.next_pc_sel.connect(self._next_pc_sel)
        self.datapath.alu_function.connect(self._alu_function)
        self.datapath.clock.connect(self.clock)
        self.datapath.reset.connect(self.reset)

        self.data_mem_if.addr.connect(self._addr)
        self.data_mem_if.wr_en.connect(self._wr_en)
        self.data_mem_if.wr_data.connect(self._wr_data)
        self.data_mem_if.rd_en.connect(self._rd_en)
        self._rd_data.connect(self.data_mem_if.rd_data)
        self.bus_addr.connect(self.data_mem_if.bus_addr)
        self.bus_wr_en.connect(self.data_mem_if.bus_wr_en)
        self.bus_wr_be.connect(self.data_mem_if.bus_wr_be)
        self.bus_wr_data.connect(self.data_mem_if.bus_wr_data)
        self.bus_rd_en.connect(self.data_mem_if.bus_rd_en)
        self.data_mem_if.bus_rd_data.connect(self.bus_rd_data)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst)
            self.ctlpath.inst_opcode.next = self.inst.value.opcode
            self.ctlpath.inst_funct3.next = self.inst.value.funct3
            self.ctlpath.inst_funct7.next = self.inst.value.funct7
            self.data_mem_if.data_format.next = self.inst.value.funct3
