"""Data Memory Bus."""

from seqlogic import Module, Op, Vec, clog2, u2bv

from . import DATA_BASE, DATA_SIZE, Addr
from .data_mem import DataMem


def f_rd_data(rd_en: Vec[1], is_data: Vec[1], data: Vec[32]) -> Vec[32]:
    sel = rd_en & is_data
    match sel:
        case "1b1":
            return data
        case _:
            return Vec[32].xprop(sel)


class DataMemBus(Module):
    """Data Memory Bus."""

    DEPTH: int = 1024

    def build(self):
        # Parameters
        word_addr_bits = clog2(self.DEPTH)
        data_start = u2bv(DATA_BASE, 32)
        data_stop = u2bv(DATA_BASE + DATA_SIZE, 32)

        # Ports
        addr = self.input(name="addr", dtype=Addr)
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_be = self.input(name="wr_be", dtype=Vec[4])
        wr_data = self.input(name="wr_data", dtype=Vec[32])
        rd_en = self.input(name="rd_en", dtype=Vec[1])
        rd_data = self.output(name="rd_data", dtype=Vec[32])
        clock = self.input(name="clock", dtype=Vec[1])

        # State
        is_data = self.logic(name="is_data", dtype=Vec[1])
        data = self.logic(name="data", dtype=Vec[32])

        # Submodules
        m, n = 2, 2 + word_addr_bits
        self.submod(
            name="data_mem",
            mod=DataMem,
            WORD_ADDR_BITS=word_addr_bits,
        ).connect(
            addr=addr[m:n],
            wr_en=(wr_en & is_data),
            wr_be=wr_be,
            wr_data=wr_data,
            rd_data=data,
            clock=clock,
        )

        self.expr(is_data, (Op.AND, (Op.GE, addr, data_start), (Op.LT, addr, data_stop)))
        self.combi(rd_data, f_rd_data, rd_en, is_data, data)
