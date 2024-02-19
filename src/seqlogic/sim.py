"""Simulation using async/await.

We are intentionally imitating the style of Python's Event Loop API:
https://docs.python.org/3/library/asyncio-eventloop.html

Credit to David Beazley's "Build Your Own Async" tutorial for inspiration:
https://www.youtube.com/watch?v=Y4Gt3Xjd7G8
"""

import heapq
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Generator
from typing import NewType, TypeAlias

Region = NewType("Region", int)


_INIT_TIME = -1
_INIT_REGION = Region(-1)
_START_TIME = 0


class SimVar:
    """The simulation component of a variable."""

    def __init__(self, value):
        """TODO(cjdrake): Write docstring."""
        self._value = self._next_value = value
        self._changed = False

        # Reference to the event loop
        self._sim = _sim

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = value

    value = property(fget=_get_value, fset=_set_value)

    def _get_next(self):
        return self._next_value

    def _set_next(self, value):
        self._changed = value != self._next_value
        self._next_value = value

        # Notify the event loop
        _sim.touch(self)

    next = property(fget=_get_next, fset=_set_next)

    def changed(self) -> bool:
        """Return True if the variable has changed."""
        return self._changed

    def dirty(self) -> bool:
        """Return True if the present state is dirty."""
        return self._next_value != self._value

    def update(self):
        """Update present state, and reset next state."""
        self._value = self._next_value
        self._changed = False


_SimQueueItem: TypeAlias = tuple[int, Region, Coroutine, SimVar | None]


class _SimQueue:
    """Priority queue for ordering task execution."""

    def __init__(self):
        self._items = []

        # Monotonically increasing integer to break ties in the heapq
        self._index: int = 0

    def __bool__(self) -> bool:
        return bool(self._items)

    def clear(self):
        self._items.clear()
        self._index = 0

    def push(self, time: int, region: Region, task: Coroutine, var: SimVar | None):
        heapq.heappush(self._items, (time, region, self._index, task, var))
        self._index += 1

    def peek(self) -> _SimQueueItem:
        time, region, _, task, var = self._items[0]
        return (time, region, task, var)

    # def pop(self) -> _SimQueueItem:
    #    time, region, _, task, var = heapq.heappop(self._items)
    #    return (time, region, task, var)

    def pop_region(self) -> Generator[_SimQueueItem, None, None]:
        time, region, _, task, var = heapq.heappop(self._items)
        yield (time, region, task, var)
        while self._items:
            t, r, _, task, var = self._items[0]
            if t == time and r == region:
                heapq.heappop(self._items)
                yield (time, region, task, var)
            else:
                break


class _SimAwaitable(Awaitable):
    """Suspend execution of the current task."""

    def __await__(self):
        var = yield
        return var


class Sim:
    """Simulation event loop."""

    def __init__(self):
        """TODO(cjdrake): Write docstring."""
        self._started: bool = False
        # Simulation time
        self._time: int = _INIT_TIME
        self._region: Region = _INIT_REGION
        # Task queue
        self._queue = _SimQueue()
        # Currently executing task
        self._task: Coroutine | None = None
        self._task_region: dict[Coroutine, Region] = {}
        # Dynamic event dependencies
        self._var2tasks: dict[SimVar, set[Coroutine]] = defaultdict(set)
        self._triggers: dict[SimVar, dict[Coroutine, Callable[[], bool]]] = defaultdict(dict)
        # Postponed actions
        self._touched_vars: set[SimVar] = set()
        # Processes
        self._procs = []

    @property
    def started(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return self._started

    def restart(self):
        """Restart the simulation."""
        self._started = False
        self._time = _INIT_TIME
        self._region = _INIT_REGION
        self._queue.clear()
        self._task = None
        self._var2tasks.clear()
        self._triggers.clear()
        self._touched_vars.clear()

    def reset(self):
        """Reset the simulation state."""
        self.restart()
        self._procs.clear()
        self._task_region.clear()

    def time(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._time

    def task(self) -> Coroutine:
        """TODO(cjdrake): Write docstring."""
        assert self._task is not None
        return self._task

    def call_soon(self, task: Coroutine, var: SimVar | None = None):
        """Schedule the task in the current timeslot."""
        region = self._task_region[task]
        self._queue.push(self._time, region, task, var)

    def call_later(self, delay: int, task: Coroutine):
        """Schedule the task after a relative delay."""
        region = self._task_region[task]
        self._queue.push(self._time + delay, region, task, None)

    def call_at(self, when: int, task: Coroutine):
        """Schedule the task for an absolute timeslot."""
        region = self._task_region[task]
        self._queue.push(when, region, task, None)

    def add_proc(self, proc, region: Region, *args, **kwargs):
        """Add a process to run at start of simulation."""
        self._procs.append((proc, region, args, kwargs))

    def add_event(self, event: Callable[[], bool]):
        """Add a conditional var => task dependency."""
        assert self._task is not None
        var = event.__self__
        self._var2tasks[var].add(self._task)
        self._triggers[var][self._task] = event

    def touch(self, var: SimVar):
        """Notify dependent tasks about a variable change."""
        tasks = [task for task in self._var2tasks[var] if self._triggers[var][task]()]
        for task in tasks:
            self.call_soon(task, var)
            self._var2tasks[var].remove(task)
            del self._triggers[var][task]

        # Add variable to update set
        self._touched_vars.add(var)

    def _limit(self, ticks: int | None, until: int | None) -> int | None:
        """Determine the run limit."""
        match ticks, until:
            # Run until no tasks left
            case None, None:
                return None
            # Run until an absolute time
            case None, int():
                return until
            # Run until a number of ticks in the future
            case int(), None:
                return max(_START_TIME, self._time) + ticks
            case _:
                s = "Expected either ticks or until to be int | None"
                raise TypeError(s)

    def _start(self):
        for proc, region, args, kwargs in self._procs:
            task = proc(*args, **kwargs)
            self._task_region[task] = region
            self.call_at(_START_TIME, task)
        self._started = True

    def run(self, ticks: int | None = None, until: int | None = None):
        """Run the simulation.

        Until:
        1. We hit the runlimit, OR
        2. There are no tasks left in the queue
        """
        limit = self._limit(ticks, until)

        # Start the simulation
        if not self._started:
            self._start()

        while self._queue:
            # Peek at when next event is scheduled
            time, region, _, _ = self._queue.peek()

            # Protect against time traveling tasks
            assert time >= self._time

            # Next task scheduled: same time slot, different region
            if time == self._time and region != self._region:
                self._region = region
            # Next task scheduled: future time slot
            elif time > self._time:
                # Update all simulation state
                self._update_vars()
                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    break
                # Otherwise, advance
                self._time = time

            # Resume execution
            for _, _, self._task, var in self._queue.pop_region():
                try:
                    self._task.send(var)
                except StopIteration:
                    pass

    def iter(
        self, ticks: int | None = None, until: int | None = None
    ) -> Generator[int, None, None]:
        """Iterate the simulation.

        Until:
        1. We hit the runlimit, OR
        2. There are no tasks left in the queue
        """
        limit = self._limit(ticks, until)

        # Start the simulation
        if not self._started:
            self._start()

        while self._queue:
            # Peek at when next event is scheduled
            time, region, _, _ = self._queue.peek()

            # Protect against time traveling tasks
            assert time >= self._time

            # Next task scheduled: same time slot, different region
            if time == self._time and region != self._region:
                self._region = region
            # Next task scheduled: future time slot
            elif time > self._time:
                # Update all simulation state
                self._update_vars()
                yield self._time
                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    return
                # Otherwise, advance
                self._time = time

            # Resume execution
            for _, _, self._task, var in self._queue.pop_region():
                try:
                    self._task.send(var)
                except StopIteration:
                    pass

    def _update_vars(self):
        """Prepare variables to enter the next time slot."""
        while self._touched_vars:
            var = self._touched_vars.pop()
            var.update()


_sim = Sim()


async def sleep(delay: int):
    """Suspend the task, and wake up after a delay."""
    _sim.call_later(delay, _sim.task())
    await _SimAwaitable()


async def notify(*events: Callable[[], bool]) -> SimVar:
    """Suspend the task, and wake up after an event notification."""
    for event in events:
        _sim.add_event(event)
    var = await _SimAwaitable()
    return var


def get_loop() -> Sim:
    """Return the event loop."""
    return _sim
