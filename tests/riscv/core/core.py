"""RiscV Core."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive

from . import AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst
from .ctl_path import CtlPath
from .data_mem_if import DataMemIf
from .data_path import DataPath


class Core(Module):
    """RiscV Core."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        bus_addr = Bits(name="bus_addr", parent=self, dtype=Vec[32])
        bus_wr_en = Bit(name="bus_wr_en", parent=self)
        bus_wr_be = Bits(name="bus_wr_be", parent=self, dtype=Vec[4])
        bus_wr_data = Bits(name="bus_wr_data", parent=self, dtype=Vec[32])
        bus_rd_en = Bit(name="bus_rd_en", parent=self)
        bus_rd_data = Bits(name="bus_rd_data", parent=self, dtype=Vec[32])

        pc = Bits(name="pc", parent=self, dtype=Vec[32])
        inst = Bits(name="inst", parent=self, dtype=Inst)

        clock = Bit(name="clock", parent=self)
        reset = Bit(name="reset", parent=self)

        # State
        pc_wr_en = Bit(name="pc_wr_en", parent=self)
        regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)
        alu_func = Bits(name="alu_func", parent=self, dtype=AluOp)
        alu_result_eq_zero = Bit(name="alu_result_eq_zero", parent=self)
        addr = Bits(name="addr", parent=self, dtype=Vec[32])
        wr_en = Bit(name="wr_en", parent=self)
        wr_data = Bits(name="wr_data", parent=self, dtype=Vec[32])
        rd_en = Bit(name="rd_en", parent=self)
        rd_data = Bits(name="rd_data", parent=self, dtype=Vec[32])

        # Submodules
        ctlpath = CtlPath(name="ctlpath", parent=self)
        self.connect(ctlpath.alu_result_eq_zero, alu_result_eq_zero)
        self.connect(pc_wr_en, ctlpath.pc_wr_en)
        self.connect(regfile_wr_en, ctlpath.regfile_wr_en)
        self.connect(alu_op_a_sel, ctlpath.alu_op_a_sel)
        self.connect(alu_op_b_sel, ctlpath.alu_op_b_sel)
        self.connect(rd_en, ctlpath.data_mem_rd_en)
        self.connect(wr_en, ctlpath.data_mem_wr_en)
        self.connect(reg_writeback_sel, ctlpath.reg_writeback_sel)
        self.connect(alu_func, ctlpath.alu_func)
        self.connect(next_pc_sel, ctlpath.next_pc_sel)

        datapath = DataPath(name="datapath", parent=self)
        self.connect(addr, datapath.data_mem_addr)
        self.connect(wr_data, datapath.data_mem_wr_data)
        self.connect(datapath.data_mem_rd_data, rd_data)
        self.connect(datapath.inst, inst)
        self.connect(pc, datapath.pc)
        self.connect(alu_result_eq_zero, datapath.alu_result_eq_zero)
        self.connect(datapath.pc_wr_en, pc_wr_en)
        self.connect(datapath.regfile_wr_en, regfile_wr_en)
        self.connect(datapath.alu_op_a_sel, alu_op_a_sel)
        self.connect(datapath.alu_op_b_sel, alu_op_b_sel)
        self.connect(datapath.reg_writeback_sel, reg_writeback_sel)
        self.connect(datapath.next_pc_sel, next_pc_sel)
        self.connect(datapath.alu_func, alu_func)
        self.connect(datapath.clock, clock)
        self.connect(datapath.reset, reset)

        data_mem_if = DataMemIf(name="data_mem_if", parent=self)
        self.connect(data_mem_if.addr, addr)
        self.connect(data_mem_if.wr_en, wr_en)
        self.connect(data_mem_if.wr_data, wr_data)
        self.connect(data_mem_if.rd_en, rd_en)
        self.connect(rd_data, data_mem_if.rd_data)
        self.connect(bus_addr, data_mem_if.bus_addr)
        self.connect(bus_wr_en, data_mem_if.bus_wr_en)
        self.connect(bus_wr_be, data_mem_if.bus_wr_be)
        self.connect(bus_wr_data, data_mem_if.bus_wr_data)
        self.connect(bus_rd_en, data_mem_if.bus_rd_en)
        self.connect(data_mem_if.bus_rd_data, bus_rd_data)

        # TODO(cjdrake): Remove
        self.bus_addr = bus_addr
        self.bus_wr_en = bus_wr_en
        self.bus_wr_be = bus_wr_be
        self.bus_wr_data = bus_wr_data
        self.bus_rd_en = bus_rd_en
        self.bus_rd_data = bus_rd_data

        self.pc = pc
        self.inst = inst

        self.clock = clock
        self.reset = reset

        self.ctlpath = ctlpath
        self.datapath = datapath
        self.data_mem_if = data_mem_if

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst)
            self.ctlpath.inst_opcode.next = self.inst.value.opcode
            self.ctlpath.inst_funct3.next = self.inst.value.funct3
            self.ctlpath.inst_funct7.next = self.inst.value.funct7
            self.data_mem_if.data_format.next = self.inst.value.funct3
