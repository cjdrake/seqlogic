"""RiscV Core."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module
from seqlogic.lbool import Vec

from . import AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst
from .ctl_path import CtlPath
from .data_mem_if import DataMemIf
from .data_path import DataPath


class Core(Module):
    """RiscV Core."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        bus_addr = self.bits(name="bus_addr", dtype=Vec[32], port=True)
        bus_wr_en = self.bit(name="bus_wr_en", port=True)
        bus_wr_be = self.bits(name="bus_wr_be", dtype=Vec[4], port=True)
        bus_wr_data = self.bits(name="bus_wr_data", dtype=Vec[32], port=True)
        bus_rd_en = self.bit(name="bus_rd_en", port=True)
        bus_rd_data = self.bits(name="bus_rd_data", dtype=Vec[32], port=True)

        pc = self.bits(name="pc", dtype=Vec[32], port=True)
        inst = self.bits(name="inst", dtype=Inst, port=True)

        clock = self.bit(name="clock", port=True)
        reset = self.bit(name="reset", port=True)

        # State
        pc_wr_en = self.bit(name="pc_wr_en")
        regfile_wr_en = self.bit(name="regfile_wr_en")
        alu_op_a_sel = self.bits(name="alu_op_a_sel", dtype=CtlAluA)
        alu_op_b_sel = self.bits(name="alu_op_b_sel", dtype=CtlAluB)
        reg_writeback_sel = self.bits(name="reg_writeback_sel", dtype=CtlWriteBack)
        next_pc_sel = self.bits(name="next_pc_sel", dtype=CtlPc)
        alu_func = self.bits(name="alu_func", dtype=AluOp)
        alu_result_eq_zero = self.bit(name="alu_result_eq_zero")
        addr = self.bits(name="addr", dtype=Vec[32])
        wr_en = self.bit(name="wr_en")
        wr_data = self.bits(name="wr_data", dtype=Vec[32])
        rd_en = self.bit(name="rd_en")
        rd_data = self.bits(name="rd_data", dtype=Vec[32])

        # Submodules
        ctlpath = self.submod(name="ctlpath", mod=CtlPath)
        self.assign(ctlpath.alu_result_eq_zero, alu_result_eq_zero)
        self.assign(pc_wr_en, ctlpath.pc_wr_en)
        self.assign(regfile_wr_en, ctlpath.regfile_wr_en)
        self.assign(alu_op_a_sel, ctlpath.alu_op_a_sel)
        self.assign(alu_op_b_sel, ctlpath.alu_op_b_sel)
        self.assign(rd_en, ctlpath.data_mem_rd_en)
        self.assign(wr_en, ctlpath.data_mem_wr_en)
        self.assign(reg_writeback_sel, ctlpath.reg_writeback_sel)
        self.assign(alu_func, ctlpath.alu_func)
        self.assign(next_pc_sel, ctlpath.next_pc_sel)

        datapath = self.submod(name="datapath", mod=DataPath)
        self.assign(addr, datapath.data_mem_addr)
        self.assign(wr_data, datapath.data_mem_wr_data)
        self.assign(datapath.data_mem_rd_data, rd_data)
        self.assign(datapath.inst, inst)
        self.assign(pc, datapath.pc)
        self.assign(alu_result_eq_zero, datapath.alu_result_eq_zero)
        self.assign(datapath.pc_wr_en, pc_wr_en)
        self.assign(datapath.regfile_wr_en, regfile_wr_en)
        self.assign(datapath.alu_op_a_sel, alu_op_a_sel)
        self.assign(datapath.alu_op_b_sel, alu_op_b_sel)
        self.assign(datapath.reg_writeback_sel, reg_writeback_sel)
        self.assign(datapath.next_pc_sel, next_pc_sel)
        self.assign(datapath.alu_func, alu_func)
        self.assign(datapath.clock, clock)
        self.assign(datapath.reset, reset)

        data_mem_if = self.submod(name="data_mem_if", mod=DataMemIf)
        self.assign(data_mem_if.addr, addr)
        self.assign(data_mem_if.wr_en, wr_en)
        self.assign(data_mem_if.wr_data, wr_data)
        self.assign(data_mem_if.rd_en, rd_en)
        self.assign(rd_data, data_mem_if.rd_data)
        self.assign(bus_addr, data_mem_if.bus_addr)
        self.assign(bus_wr_en, data_mem_if.bus_wr_en)
        self.assign(bus_wr_be, data_mem_if.bus_wr_be)
        self.assign(bus_wr_data, data_mem_if.bus_wr_data)
        self.assign(bus_rd_en, data_mem_if.bus_rd_en)
        self.assign(data_mem_if.bus_rd_data, bus_rd_data)

        # Combinational Logic
        ys = (
            ctlpath.inst_opcode,
            ctlpath.inst_funct3,
            ctlpath.inst_funct7,
            data_mem_if.data_format,
        )
        self.combi(ys, lambda x: (x.opcode, x.funct3, x.funct7, x.funct3), inst)
