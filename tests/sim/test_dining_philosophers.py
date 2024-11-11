"""Test dining philosophers."""

from enum import Enum, auto
from random import randint

from seqlogic import Lock, create_task, now, run, sleep

# Number of philosophers
N = 5


# Philosopher state
class State(Enum):
    INITIAL = auto()
    THINKING = auto()
    HUNGRY = auto()
    EATING = auto()


# Eat [min, max] time
EAT_TICKS = (50, 100)

# Think [min, max] time
THINK_TICKS = (50, 100)

# Simulation time
T = 1000


# Optional mask to filter print output
# If pmask & (1<<i), print philosopher state updates.
_pmask = (1 << N) - 1


# Philosophers and Forks
state = [State.INITIAL for _ in range(N)]
forks = [Lock() for _ in range(N)]


def init(pmask: int = (1 << N) - 1):
    """Initialize all philosophers and forks."""
    global _pmask, state, forks
    _pmask = pmask
    state = [State.INITIAL for _ in range(N)]
    forks = [Lock() for _ in range(N)]


def _update(i: int, ns: State):
    """Update philosopher[i] state."""
    if _pmask & (1 << i):
        print(f"[{now():08}] P{i} {state[i].name:8} => {ns.name:8}")
    state[i] = ns


async def think(i: int):
    """Philosopher[i] thinks for a random amount of time."""
    _update(i, State.THINKING)
    await sleep(randint(*THINK_TICKS))


async def pick_up_forks(i: int):
    """Philosopher[i] is hungry. Pick up left/right forks."""
    _update(i, State.HUNGRY)

    # Wait on forks in (left, right) order
    first, second = i, (i + 1) % N

    while True:
        # Wait until first fork is available
        await forks[first].acquire()

        # If second fork is available, get it.
        if forks[second].try_acquire():
            break

        # Second fork is NOT available:
        # 1. Release the first fork
        forks[first].release()
        # 2. Swap which fork we're waiting on first
        first, second = second, first


async def eat(i: int):
    """Philosopher[i] eats for a random amount of time."""
    _update(i, State.EATING)
    await sleep(randint(*EAT_TICKS))


def put_down_forks(i: int):
    """Philosopher[i] is not hungry. Put down left/right forks."""
    first, second = i, (i + 1) % N
    forks[first].release()
    forks[second].release()


async def philosopher(i: int):
    while True:
        await think(i)
        await pick_up_forks(i)
        await eat(i)
        put_down_forks(i)


async def main():
    for i in range(N):
        create_task(philosopher(i))


def test_dp(capsys):
    """This is a random algorithm, so we're only doing some basic checks."""
    init()
    run(main(), until=T)
    s = capsys.readouterr().out
    for i in range(N):
        # Verify all philosophers did at least one full cycle
        assert f"P{i} INITIAL  => THINKING" in s
        assert f"P{i} THINKING => HUNGRY" in s
        assert f"P{i} HUNGRY   => EATING" in s
        assert f"P{i} EATING   => THINKING" in s
