"""Data Memory Bus."""

# pyright: reportAttributeAccessIssue=false

import operator

from seqlogic import Module, clog2
from seqlogic.vec import Vec, uint2vec

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

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)

        # Parameters
        word_addr_bits = clog2(depth)
        data_start = DATA_BASE
        data_stop = DATA_BASE + DATA_SIZE

        # Ports
        addr = self.input(name="addr", dtype=Addr)
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_be = self.input(name="wr_be", dtype=Vec[4])
        wr_data = self.input(name="wr_data", dtype=Vec[32])
        rd_en = self.input(name="rd_en", dtype=Vec[1])
        rd_data = self.output(name="rd_data", dtype=Vec[32])
        clock = self.input(name="clock", dtype=Vec[1])

        # State
        is_data = self.bit(name="is_data")
        data = self.bits(name="data", dtype=Vec[32])

        # Submodules
        m, n = 2, 2 + word_addr_bits
        self.submod(
            name="data_mem",
            mod=DataMem,
            word_addr_bits=word_addr_bits,
        ).connect(
            addr=(lambda a: a[m:n], addr),
            wr_en=(operator.and_, wr_en, is_data),
            wr_be=wr_be,
            wr_data=wr_data,
            rd_data=data,
            clock=clock,
        )

        # Combinational Logic
        def f_is_data(addr: Addr) -> Vec[1]:
            start = uint2vec(data_start, 32)
            stop = uint2vec(data_stop, 32)
            return start.leu(addr) & addr.ltu(stop)

        self.combi(is_data, f_is_data, addr)
        self.combi(rd_data, f_rd_data, rd_en, is_data, data)
