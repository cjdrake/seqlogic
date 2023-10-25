"""
Simulation using async/await

We are intentionally imitating the style of Python's Event Loop API:
https://docs.python.org/3/library/asyncio-eventloop.html

Credit to David Beazley's "Build Your Own Async" tutorial for inspiration:
https://www.youtube.com/watch?v=Y4Gt3Xjd7G8
"""


import heapq
from collections.abc import Awaitable, Callable, Coroutine, Generator
from enum import Enum
from typing import NewType, Optional, TypeAlias

import networkx as nx

_Event: TypeAlias = Callable[[], bool]
_Task: TypeAlias = Coroutine
_Proc: TypeAlias = Callable[[], _Task]

_Region = NewType("Region", int)
_Time = NewType("Time", int)


_INIT_TIME = _Time(-1)
_INIT_REGION = _Region(-1)
_START_TIME = _Time(0)


class State(Enum):
    INVALID = 0b00
    CLEAN = 0b01
    DIRTY = 0b11


class SimVar:
    """
    The simulation component of a variable.
    """

    def __init__(self, value):
        self._value = value
        self._next = None
        self._state: State = State.INVALID

        # Reference to the event loop
        self._sim = _sim

    @property
    def value(self):
        return self._value

    def _set_next(self, value):
        # Protect against double assignment in the same time slot
        assert self._state is State.INVALID

        self._next = value
        if value != self._value:
            self._state = State.DIRTY
        else:
            self._state = State.CLEAN

        _sim.notify(self)

    next = property(fset=_set_next)

    def dirty(self) -> bool:
        return self._state is State.DIRTY

    def update(self):
        self._value = self._next
        self._next = None
        self._state = State.INVALID


_SimQueueItem = tuple[_Time, _Region, _Task, Optional[SimVar]]


class _SimQueue:
    """
    Priority queue for ordering task execution
    """

    def __init__(self):
        self._items = []

        # Monotonically increasing integer to break ties in the heapq
        self._index: int = 0

    def __bool__(self) -> bool:
        return bool(self._items)

    def clear(self):
        self._items.clear()
        self._index = 0

    def push(self, time: _Time, region: _Region, task: _Task, var: Optional[SimVar] = None):
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
    """
    Suspend execution of the current task.
    """

    def __await__(self):
        var = yield
        return var


class Sim:
    """
    Simulation event loop
    """

    def __init__(self):
        self._started: bool = False
        # Simulation time
        self._time: _Time = _INIT_TIME
        self._region: _Region = _INIT_REGION
        # Task queue
        self._queue = _SimQueue()
        # Currently executing task
        self._task: Optional[_Task] = None
        self._task_region: dict[_Task, _Region] = {}
        # Dynamic event dependencies
        self._deps = nx.DiGraph()
        # Postponed actions
        self._valid_vars: set[SimVar] = set()
        # Processes
        self._procs = []

    @property
    def started(self) -> bool:
        return self._started

    def restart(self):
        """Restart the simulation."""
        self._started = False
        self._time = _INIT_TIME
        self._region = _INIT_REGION
        self._queue.clear()
        self._task = None
        self._deps.clear()
        self._valid_vars.clear()

    def reset(self):
        """Reset the simulation state."""
        self.restart()
        self._procs.clear()
        self._task_region.clear()

    def time(self) -> _Time:
        return self._time

    def task(self) -> _Task:
        assert self._task is not None
        return self._task

    def call_soon(self, task: _Task, var: Optional[SimVar] = None):
        """Schedule the task in the current timeslot."""
        region = self._task_region[task]
        self._queue.push(self._time, region, task, var)

    def call_later(self, delay: _Time, task: _Task):
        """Schedule the task after a relative delay."""
        region = self._task_region[task]
        self._queue.push(_Time(self._time + delay), region, task)

    def call_at(self, when: _Time, task: _Task):
        """Schedule the task for an absolute timeslot."""
        region = self._task_region[task]
        self._queue.push(when, region, task)

    def add_proc(self, proc: _Proc, region: _Region, *args, **kwargs):
        """Add a process to run at start of simulation."""
        self._procs.append((proc, region, args, kwargs))

    def add_event(self, event: _Event):
        """Add a conditional var => task dependency."""
        var = event.__self__
        self._deps.add_edge(var, event)
        self._deps.add_edge(event, self._task)

    def _prune(self, u, v):
        """Prune (u, v) from the dependency graph."""
        self._deps.remove_edge(u, v)
        if self._deps.in_degree(v) == 0:
            self._deps.remove_node(v)

    def notify(self, var: SimVar):
        """Notify dependent tasks about a variable change."""
        if var in self._deps and var.dirty():
            notifications = {e: set(self._deps[e]) for e in self._deps[var] if e()}
            for event, tasks in notifications.items():
                for task in tasks:
                    self.call_soon(task, var)
                    self._prune(event, task)
                self._prune(var, event)

        # Add variable to update set
        self._valid_vars.add(var)

    def _limit(
        self, ticks: Optional[_Time] = None, until: Optional[_Time] = None
    ) -> Optional[_Time]:
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
                return _Time(max(_START_TIME, self._time) + ticks)
            case _:
                s = "Expected either ticks or until to be int | None"
                raise TypeError(s)

    def _start(self):
        for proc, region, args, kwargs in self._procs:
            task = proc(*args, **kwargs)
            self._task_region[task] = region
            self.call_at(_START_TIME, task)
        self._started = True

    def run(self, ticks: Optional[_Time] = None, until: Optional[_Time] = None):
        """
        Run the simulation until:
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

    def _update_vars(self):
        """Prepare variables to enter the next time slot."""
        while self._valid_vars:
            self._valid_vars.pop().update()


_sim = Sim()


async def sleep(delay: _Time):
    """Suspend the task, and wake up after a delay."""
    _sim.call_later(delay, _sim.task())
    await _SimAwaitable()


async def notify(*events: _Event) -> SimVar:
    """Suspend the task, and wake up after an event notification."""
    for event in events:
        _sim.add_event(event)
    var = await _SimAwaitable()
    return var


def get_loop() -> Sim:
    """Return the event loop."""
    return _sim
