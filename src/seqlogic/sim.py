"""Simulation using async/await.

We are intentionally imitating the style of Python's Event Loop API:
https://docs.python.org/3/library/asyncio-eventloop.html

Credit to David Beazley's "Build Your Own Async" tutorial for inspiration:
https://www.youtube.com/watch?v=Y4Gt3Xjd7G8
"""

import heapq
import inspect
from abc import ABC
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Generator, Hashable
from enum import IntEnum, auto
from functools import partial
from typing import TypeAlias

_INIT_TIME = -1
_START_TIME = 0


class Region(IntEnum):
    REACTIVE = auto()
    ACTIVE = auto()


class State(ABC):
    """Model component."""

    def __init__(self):
        # Reference to the event loop
        self._sim = _sim

    def get_value(self):
        raise NotImplementedError()

    value = property(fget=get_value)

    def changed(self) -> bool:
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()


class Singular(State):
    """Model state organized as a single unit."""

    def __init__(self, value):
        super().__init__()
        self._value = value
        self._next_value = value
        self._changed = False

    def get_value(self):
        if self._sim.region == Region.REACTIVE:
            return self._next_value
        return self._value

    value = property(fget=get_value)

    def set_next(self, value):
        self._changed = value != self._next_value
        self._next_value = value

        # Notify the event loop
        _sim.touch(self)

    next = property(fset=set_next)

    def changed(self) -> bool:
        return self._changed

    def update(self):
        self._value = self._next_value
        self._changed = False

    def dirty(self) -> bool:
        return self._next_value != self._value


class Aggregate(State):
    """Model state organized as multiple units."""

    def __init__(self, value):
        super().__init__()
        self._value = defaultdict(lambda: value)
        self._next_value = defaultdict(lambda: value)
        self._changed: set[Hashable] = set()

    def __getitem__(self, key: Hashable):
        return _AggrItem(self, key)

    def get_value(self):
        if self._sim.region == Region.REACTIVE:
            return self._next_value.copy()
        return self._value.copy()

    value = property(fget=get_value)

    def set_next(self, key: Hashable, value):
        if value != self._next_value[key]:
            self._changed.add(key)
        self._next_value[key] = value

        # Notify the event loop
        _sim.touch(self)

    def changed(self) -> bool:
        return bool(self._changed)

    def update(self):
        for key in self._changed:
            self._value[key] = self._next_value[key]
        self._changed.clear()


class _AggrItem:
    """Wrap Aggregate item getter/setter."""

    def __init__(self, aggr: Aggregate, key: Hashable):
        self._aggr = aggr
        self._key = key

    def get_value(self):
        return self._aggr.value[self._key]

    value = property(fget=get_value)

    def set_next(self, value):
        self._aggr.set_next(self._key, value)

    next = property(fset=set_next)


_SimQueueItem: TypeAlias = tuple[int, Region, Coroutine, State | None]


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

    def push(self, time: int, region: Region, task: Coroutine, state: State | None = None):
        heapq.heappush(self._items, (time, region, self._index, task, state))
        self._index += 1

    def peek(self) -> _SimQueueItem:
        time, region, _, task, state = self._items[0]
        return (time, region, task, state)

    # def pop(self) -> _SimQueueItem:
    #    time, region, _, task, state = heapq.heappop(self._items)
    #    return (time, region, task, state)

    def pop_region(self) -> Generator[_SimQueueItem, None, None]:
        time, region, _, task, state = heapq.heappop(self._items)
        yield (time, region, task, state)
        while self._items:
            t, r, _, task, state = self._items[0]
            if t == time and r == region:
                heapq.heappop(self._items)
                yield (time, region, task, state)
            else:
                break


class SimAwaitable(Awaitable):
    """Suspend execution of the current task."""

    def __await__(self):
        state = yield
        return state


class Sim:
    """Simulation event loop."""

    def __init__(self):
        """Initialize simulation."""
        self._started: bool = False
        # Simulation time
        self._time: int = _INIT_TIME
        self._region: Region | None = None
        # Task queue
        self._queue = _SimQueue()
        # Currently executing task
        self._task: Coroutine | None = None
        self._task_region: dict[Coroutine, Region] = {}
        # Dynamic event dependencies
        self._waiting: dict[State, set[Coroutine]] = defaultdict(set)
        self._triggers: dict[State, dict[Coroutine, Callable[[], bool]]] = defaultdict(dict)
        # Postponed actions
        self._touched: set[State] = set()
        # Processes
        self._procs = []

    @property
    def started(self) -> bool:
        return self._started

    def restart(self):
        """Restart simulation."""
        self._started = False
        self._time = _INIT_TIME
        self._region = None
        self._queue.clear()
        self._task = None
        self._waiting.clear()
        self._triggers.clear()
        self._touched.clear()

    def reset(self):
        """Reset simulation state."""
        self.restart()
        self._procs.clear()
        self._task_region.clear()

    @property
    def time(self) -> int:
        return self._time

    @property
    def region(self) -> Region | None:
        return self._region

    @property
    def task(self) -> Coroutine | None:
        return self._task

    def call_soon(self, task: Coroutine, state: State | None = None):
        """Schedule task in the current timeslot."""
        region = self._task_region[task]
        self._queue.push(self._time, region, task, state)

    def call_later(self, delay: int, task: Coroutine):
        """Schedule task after a relative delay."""
        region = self._task_region[task]
        self._queue.push(self._time + delay, region, task)

    def call_at(self, when: int, task: Coroutine):
        """Schedule task for an absolute timeslot."""
        region = self._task_region[task]
        self._queue.push(when, region, task)

    def add_proc(self, region: Region, func, *args, **kwargs):
        """Add a process to run at start of simulation."""
        self._procs.append((region, func, args, kwargs))

    def add_event(self, state: State, cond: Callable[[], bool]):
        """Add a conditional state => task dependency."""
        assert self._task is not None
        self._waiting[state].add(self._task)
        self._triggers[state][self._task] = cond

    def touch(self, state: State):
        """Notify dependent tasks about state change."""
        tasks = [task for task in self._waiting[state] if self._triggers[state][task]()]
        for task in tasks:
            self.call_soon(task, state)
            self._waiting[state].remove(task)
            del self._triggers[state][task]

        # Add state to update set
        self._touched.add(state)

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
        for region, func, args, kwargs in self._procs:
            task = func(*args, **kwargs)
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
                self._update_state()

                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    break
                # Otherwise, advance
                self._time = time
                self._region = region

            # Resume execution
            for _, _, self._task, state in self._queue.pop_region():
                try:
                    self._task.send(state)
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
                self._update_state()
                yield self._time

                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    return
                # Otherwise, advance
                self._time = time
                self._region = region

            # Resume execution
            for _, _, self._task, state in self._queue.pop_region():
                try:
                    self._task.send(state)
                except StopIteration:
                    pass

    def _update_state(self):
        """Prepare state to enter the next time slot."""
        while self._touched:
            state = self._touched.pop()
            state.update()


_sim = Sim()


async def sleep(delay: int):
    """Suspend the task, and wake up after a delay."""
    assert _sim.task is not None
    _sim.call_later(delay, _sim.task)
    await SimAwaitable()


async def changed(*states: State) -> State:
    """Resume execution upon state change."""
    for state in states:
        _sim.add_event(state, state.changed)
    state = await SimAwaitable()
    return state


async def resume(*events: tuple[State, Callable[[], bool]]) -> State:
    """Resume execution upon event."""
    for state, cond in events:
        _sim.add_event(state, cond)
    state = await SimAwaitable()
    return state


def get_loop() -> Sim:
    """Return the event loop."""
    return _sim


class _Schedule:
    """Add scheduling semantics to coroutine functions."""

    def __init__(self, region: Region, func):
        self._region = region
        assert inspect.iscoroutinefunction(func)
        self._func = func

    def __get__(self, obj, cls=None):
        return self._region, partial(self._func, obj)


class active(_Schedule):
    """Decorate a coroutine to run during active scheduling region."""

    def __init__(self, func):
        super().__init__(Region.ACTIVE, func)


class reactive(_Schedule):
    """Decorate a coroutine to run during reactive scheduling region."""

    def __init__(self, func):
        super().__init__(Region.REACTIVE, func)
