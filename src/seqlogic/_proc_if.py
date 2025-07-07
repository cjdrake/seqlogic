"""Process Interface"""

from deltacycle import TaskCoro


class ProcIf:
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._reactive: list[TaskCoro] = []
        self._active: list[TaskCoro] = []
        self._inactive: list[TaskCoro] = []

    @property
    def reactive(self) -> list[TaskCoro]:
        return self._reactive

    @property
    def active(self) -> list[TaskCoro]:
        return self._active

    @property
    def inactive(self) -> list[TaskCoro]:
        return self._inactive
