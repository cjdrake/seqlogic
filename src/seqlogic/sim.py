"""Simulation using async/await.

We are intentionally imitating the style of Python's Event Loop API:
https://docs.python.org/3/library/asyncio-eventloop.html

Credit to David Beazley's "Build Your Own Async" tutorial for inspiration:
https://www.youtube.com/watch?v=Y4Gt3Xjd7G8
"""

import heapq
from abc import ABC
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Generator, Hashable
from enum import IntEnum, auto

_INIT_TIME = -1
_START_TIME = 0

type Trigger = Callable[[], bool]


class Region(IntEnum):
    REACTIVE = auto()
    ACTIVE = auto()


class State(ABC):
    """Model component."""

    def __init__(self):
        # Reference to the event loop
        self._sim = _sim

    state = property(fget=NotImplemented)

    def changed(self) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def update(self):
        raise NotImplementedError()  # pragma: no cover


class Value(ABC):
    """State value."""

    def _get_value(self):
        raise NotImplementedError()  # pragma: no cover

    value = property(fget=_get_value)

    def _set_next(self, value):
        raise NotImplementedError()  # pragma: no cover

    next = property(fset=_set_next)


class Singular(State, Value):
    """Model state organized as a single unit."""

    def __init__(self, value):
        super().__init__()
        self._value = value
        self._next_value = value
        self._changed = False

    def _get_value(self):
        if self._sim.region() == Region.REACTIVE:
            return self._next_value
        return self._value

    value = property(fget=_get_value)
    state = property(fget=_get_value)

    def _set_next(self, value):
        self._changed = value != self._next_value
        self._next_value = value

        # Notify the event loop
        _sim.touch(self)

    next = property(fset=_set_next)

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
        self._values = defaultdict(lambda: value)
        self._next_values = defaultdict(lambda: value)
        self._changed: set[Hashable] = set()

    def __getitem__(self, key: Hashable):
        def fget():
            values = self._get_values()
            return values[key]

        def fset(value):
            self._set_next(key, value)

        return _AggrValue(fget, fset)

    def _get_values(self):
        if self._sim.region() == Region.REACTIVE:
            return self._next_values.copy()
        return self._values.copy()

    values = property(fget=_get_values)
    state = property(fget=_get_values)

    def _set_next(self, key: Hashable, value):
        if value != self._next_values[key]:
            self._changed.add(key)
        self._next_values[key] = value

        # Notify the event loop
        _sim.touch(self)

    def changed(self) -> bool:
        return bool(self._changed)

    def update(self):
        for key in self._changed:
            self._values[key] = self._next_values[key]
        self._changed.clear()


class _AggrValue(Value):
    """Wrap Aggregate value getter/setter."""

    def __init__(self, fget, fset):
        self._fget = fget
        self._fset = fset

    def _get_value(self):
        return self._fget()

    value = property(fget=_get_value)

    def _set_next(self, value):
        self._fset(value)

    next = property(fset=_set_next)


type _SimQueueItem = tuple[int, Region, Coroutine, State | None]


class _SimQueue:
    """Priority queue for ordering task execution."""

    def __init__(self):
        self._items: list[_SimQueueItem] = []

        # Monotonically increasing integer to break ties in the heapq
        self._index: int = 0

    def __bool__(self) -> bool:
        return bool(self._items)

    def clear(self):
        self._items.clear()
        self._index = 0

    def push(self, time: int, region: Region, coro: Coroutine, state: State | None = None):
        heapq.heappush(self._items, (time, region, self._index, coro, state))
        self._index += 1

    def peek(self) -> _SimQueueItem:
        time, region, _, coro, state = self._items[0]
        return (time, region, coro, state)

    # def pop(self) -> _SimQueueItem:
    #    time, region, _, coro, state = heapq.heappop(self._items)
    #    return (time, region, coro, state)

    def pop_region(self) -> Generator[_SimQueueItem, None, None]:
        time, region, _, coro, state = heapq.heappop(self._items)
        yield (time, region, coro, state)
        while self._items:
            t, r, _, coro, state = self._items[0]
            if t == time and r == region:
                heapq.heappop(self._items)
                yield (time, region, coro, state)
            else:
                break


class SimAwaitable(Awaitable):
    """Suspend execution of the current task."""

    def __await__(self):
        # Suspend
        state = yield
        # Resume
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
        self._coro: Coroutine | None = None
        # Dynamic event dependencies
        self._waiting: dict[State, set[tuple[Region, Coroutine]]] = defaultdict(set)
        self._triggers: dict[State, dict[Coroutine, Trigger]] = defaultdict(dict)
        # Postponed actions
        self._touched: set[State] = set()
        # Processes
        self._initial: list[tuple[Region, Coroutine]] = []

    @property
    def started(self) -> bool:
        return self._started

    def restart(self):
        """Restart simulation."""
        self._started = False
        self._time = _INIT_TIME
        self._region = None
        self._queue.clear()
        self._coro = None
        self._waiting.clear()
        self._triggers.clear()
        self._touched.clear()

    def reset(self):
        """Reset simulation state."""
        self.restart()
        self._initial.clear()

    def time(self) -> int:
        return self._time

    def region(self) -> Region:
        return self._region

    def coro(self) -> Coroutine:
        return self._coro

    def add_initial(self, region: Region, coro: Coroutine):
        """Add a task to run at start of simulation."""
        self._initial.append((region, coro))

    def add_active(self, coro: Coroutine):
        """Add a coroutine in active region to run at start of simulation."""
        self._initial.append((Region.ACTIVE, coro))

    def set_timer(self, delay: int):
        """Schedule current coroutine after delay."""
        self._queue.push(self._time + delay, self._region, self._coro)

    def set_trigger(self, state: State, trigger: Trigger):
        """Schedule current coroutine after a state update trigger."""
        self._waiting[state].add((self._region, self._coro))
        self._triggers[state][self._coro] = trigger

    def touch(self, state: State):
        """Schedule coroutines triggered by touching model state."""
        waiting = self._waiting[state]
        triggers = self._triggers[state]
        pending = [(r, c) for (r, c) in waiting if triggers[c]()]
        for region, coro in pending:
            self._queue.push(self._time, region, coro, state)
            self._waiting[state].remove((region, coro))
            del self._triggers[state][coro]
        # Add state to update set
        self._touched.add(state)

    def _update(self):
        while self._touched:
            state = self._touched.pop()
            state.update()

    def _advance(self, time: int, region: Region):
        self._time = time
        self._region = region

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
        for region, coro in self._initial:
            self._queue.push(_START_TIME, region, coro)
        self._started = True

    def _run_kernel(self, limit: int | None):
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
                # Update simulation state
                self._update()

                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    break
                # Otherwise, advance to new timeslot
                self._advance(time, region)

            # Resume execution
            for _, _, self._coro, state in self._queue.pop_region():
                try:
                    self._coro.send(state)
                except StopIteration:
                    pass

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

        self._run_kernel(limit)

    def _iter_kernel(self, limit: int | None) -> Generator[int, None, None]:
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
                # Update simulation state
                self._update()
                yield self._time

                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    return
                # Otherwise, advance to new timeslot
                self._advance(time, region)

            # Resume execution
            for _, _, self._coro, state in self._queue.pop_region():
                try:
                    self._coro.send(state)
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

        yield from self._iter_kernel(limit)


_sim = Sim()


async def sleep(delay: int):
    """Suspend the task, and wake up after a delay."""
    _sim.set_timer(delay)
    await SimAwaitable()


async def changed(*states: State) -> State:
    """Resume execution upon state change."""
    for state in states:
        _sim.set_trigger(state, state.changed)
    state = await SimAwaitable()
    return state


async def resume(*events: tuple[State, Trigger]) -> State:
    """Resume execution upon event."""
    for state, trigger in events:
        _sim.set_trigger(state, trigger)
    state = await SimAwaitable()
    return state


def get_loop() -> Sim:
    """Return the event loop."""
    return _sim


class ProcIf(ABC):
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._initial: list[tuple[Region, Coroutine]] = []

    def add_initial(self):
        for region, coro in self._initial:
            _sim.add_initial(region, coro)
