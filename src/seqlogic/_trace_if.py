"""Trace Interface"""

from collections import defaultdict

from vcd.writer import VCDWriter as VcdWriter


class TraceIf:
    """Tracing interface.

    Implemented by components that support debug dump.
    """

    def dump_waves(self, waves: defaultdict[int, dict], pattern: str):
        """Dump design elements w/ names matching pattern to waves dict."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """Dump design elements w/ names matching pattern to VCD file."""
