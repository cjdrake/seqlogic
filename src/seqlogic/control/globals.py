"""Globals: clocks and resets."""

from ..design import Packed
from ..sim import sleep


async def drive_reset(
    y: Packed,
    pos: bool = False,
    offticks: int = 0,
    onticks: int = 1,
):
    r"""
    Drive a reset signal.

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾

    Args:

    Raises:
        ValueError: Invalid parameter values
    """
    if offticks < 0:
        raise ValueError(f"Expected offticks ≥ 0, got {offticks}")
    if onticks < 0:
        raise ValueError(f"Expected onticks ≥ 0, got {onticks}")

    # T = 0
    y.next = "1b0" if pos else "1b1"
    await sleep(offticks)

    y.next = ~y.value
    await sleep(onticks)

    y.next = ~y.value


async def drive_clock(
    y: Packed,
    pos: bool = True,
    shiftticks: int = 0,
    onticks: int = 1,
    offticks: int = 1,
):
    r"""
    Drive a clock signal.

    ________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________

    Args:

    The period is onticks + offticks
    The duty cycle is onticks / (onticks + offticks)

    Raises:
        ValueError: Invalid parameter values
    """
    if shiftticks < 0:
        raise ValueError(f"Expected shiftticks ≥ 0, got {shiftticks}")
    if onticks < 0:
        raise ValueError(f"Expected onticks ≥ 0, got {onticks}")
    if offticks < 0:
        raise ValueError(f"Expected offticks ≥ 0, got {offticks}")

    # T = 0
    y.next = "1b0" if pos else "1b1"
    await sleep(shiftticks)

    while True:
        y.next = ~y.value
        await sleep(onticks)

        y.next = ~y.value
        await sleep(offticks)
