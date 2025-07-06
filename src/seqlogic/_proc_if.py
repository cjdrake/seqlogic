"""Process Interface"""

from collections.abc import Coroutine


class ProcIf:
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._reactive: list[Coroutine] = []
        self._active: list[Coroutine] = []
        self._inactive: list[Coroutine] = []

    @property
    def reactive(self) -> list[Coroutine]:
        return self._reactive

    @property
    def active(self) -> list[Coroutine]:
        return self._active

    @property
    def inactive(self) -> list[Coroutine]:
        return self._inactive
