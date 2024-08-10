"""RiscV Core."""

from seqlogic import Module, Vec

from . import Addr, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst
from .ctl_path import CtlPath
from .data_mem_if import DataMemIf
from .data_path import DataPath


class Core(Module):
    """RiscV Core."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        bus_addr = self.output(name="bus_addr", dtype=Addr)
        bus_wr_en = self.output(name="bus_wr_en", dtype=Vec[1])
        bus_wr_be = self.output(name="bus_wr_be", dtype=Vec[4])
        bus_wr_data = self.output(name="bus_wr_data", dtype=Vec[32])
        bus_rd_en = self.output(name="bus_rd_en", dtype=Vec[1])
        bus_rd_data = self.input(name="bus_rd_data", dtype=Vec[32])

        pc = self.output(name="pc", dtype=Vec[32])
        inst = self.input(name="inst", dtype=Inst)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        # State
        pc_wr_en = self.logic(name="pc_wr_en", dtype=Vec[1])
        reg_wr_en = self.logic(name="reg_wr_en", dtype=Vec[1])
        alu_op_a_sel = self.logic(name="alu_op_a_sel", dtype=CtlAluA)
        alu_op_b_sel = self.logic(name="alu_op_b_sel", dtype=CtlAluB)
        next_pc_sel = self.logic(name="next_pc_sel", dtype=CtlPc)
        alu_op = self.logic(name="alu_op", dtype=AluOp)
        alu_result_eq_zero = self.logic(name="alu_result_eq_zero", dtype=Vec[1])
        reg_wr_sel = self.logic(name="reg_wr_sel", dtype=CtlWriteBack)

        addr = self.logic(name="addr", dtype=Addr)
        wr_en = self.logic(name="wr_en", dtype=Vec[1])
        wr_data = self.logic(name="wr_data", dtype=Vec[32])
        rd_en = self.logic(name="rd_en", dtype=Vec[1])
        rd_data = self.logic(name="rd_data", dtype=Vec[32])

        # Submodules
        self.submod(
            name="ctlpath",
            mod=CtlPath,
        ).connect(
            opcode=inst.getattr("opcode"),
            funct3=inst.getattr("funct3"),
            funct7=inst.getattr("funct7"),
            alu_result_eq_zero=alu_result_eq_zero,
            pc_wr_en=pc_wr_en,
            reg_wr_en=reg_wr_en,
            alu_op_a_sel=alu_op_a_sel,
            alu_op_b_sel=alu_op_b_sel,
            data_mem_rd_en=rd_en,
            data_mem_wr_en=wr_en,
            reg_wr_sel=reg_wr_sel,
            alu_op=alu_op,
            next_pc_sel=next_pc_sel,
        )

        self.submod(
            name="datapath",
            mod=DataPath,
        ).connect(
            data_mem_addr=addr,
            data_mem_wr_data=wr_data,
            data_mem_rd_data=rd_data,
            inst=inst,
            pc=pc,
            alu_result_eq_zero=alu_result_eq_zero,
            pc_wr_en=pc_wr_en,
            reg_wr_en=reg_wr_en,
            alu_op_a_sel=alu_op_a_sel,
            alu_op_b_sel=alu_op_b_sel,
            alu_op=alu_op,
            reg_wr_sel=reg_wr_sel,
            next_pc_sel=next_pc_sel,
            clock=clock,
            reset=reset,
        )

        self.submod(
            name="data_mem_if",
            mod=DataMemIf,
        ).connect(
            data_format=inst.getattr("funct3"),
            addr=addr,
            wr_en=wr_en,
            wr_data=wr_data,
            rd_en=rd_en,
            rd_data=rd_data,
            bus_addr=bus_addr,
            bus_wr_en=bus_wr_en,
            bus_wr_be=bus_wr_be,
            bus_wr_data=bus_wr_data,
            bus_rd_en=bus_rd_en,
            bus_rd_data=bus_rd_data,
        )
