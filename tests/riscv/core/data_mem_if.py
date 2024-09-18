"""Data Memory Interface."""

from seqlogic import Cat, Module, Mux, Vec, cat, rep

from . import Addr


def f_rd_data(data_format: Vec[3], bus_rd_data: Vec[32], byte_addr: Vec[2]) -> Vec[32]:
    data = bus_rd_data >> cat("3b000", byte_addr)
    sel = data_format[:2]
    match sel:
        case "2b00":
            pad = ~data_format[2] & data[8 - 1]
            return cat(data[:8], rep(pad, 24))
        case "2b01":
            pad = ~data_format[2] & data[16 - 1]
            return cat(data[:16], rep(pad, 16))
        case "2b10":
            return data
        case _:
            return Vec[32].xprop(sel)


class DataMemIf(Module):
    """Data Memory Interface."""

    def build(self):
        # Ports
        data_format = self.input(name="data_format", dtype=Vec[3])

        addr = self.input(name="addr", dtype=Addr)
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_data = self.input(name="wr_data", dtype=Vec[32])
        rd_en = self.input(name="rd_en", dtype=Vec[1])
        rd_data = self.output(name="rd_data", dtype=Vec[32])

        bus_addr = self.output(name="bus_addr", dtype=Addr)
        bus_wr_en = self.output(name="bus_wr_en", dtype=Vec[1])
        bus_wr_be = self.output(name="bus_wr_be", dtype=Vec[4])
        bus_wr_data = self.output(name="bus_wr_data", dtype=Vec[32])
        bus_rd_en = self.output(name="bus_rd_en", dtype=Vec[1])
        bus_rd_data = self.input(name="bus_rd_data", dtype=Vec[32])

        byte_addr = self.logic(name="byte_addr", dtype=Vec[2])

        self.assign(bus_addr, addr)
        self.assign(bus_wr_en, wr_en)
        self.assign(bus_rd_en, rd_en)

        # Combinational Logic
        self.expr(
            bus_wr_be,
            Mux(
                data_format[:2],
                x0=("4b0001" << byte_addr),
                x1=("4b0011" << byte_addr),
                x2=("4b1111" << byte_addr),
                x3="4b0000",
            ),
        )
        self.expr(bus_wr_data, wr_data << Cat("3b000", byte_addr))
        self.combi(rd_data, f_rd_data, data_format, bus_rd_data, byte_addr)
        self.expr(byte_addr, addr[:2])
