#!/usr/bin/env python3
"""Holding a generator's colour draws, shared by whoever needs an episode.

The rollout builds an episode by holding one colour assignment across its four
pairs, and anything that wants to judge a maker the way the rollout uses it has
to build episodes the same way. Keeping this in one place is not tidiness: the
scorer grew its own idea of what a demonstration was -- three pairs pooled
across seeds, each in its own palette -- and marked four makers down to a third
of their coverage for failing to read a convention those pairs never shared.
"""
from __future__ import annotations

import collections
import os
import linecache
import random as _random
import re
import sys

PIPELINE_DIR = __file__.rsplit("/", 1)[0]


COLOUR_ARGS = re.compile(r"(?:choice|sample)\(\s*([A-Za-z_]\w*)")
COLOUR_NAME = re.compile(r"col|^itv$|^remitv$", re.I)


_COLOUR_SITE = {}
# How many times a generator may restart its colour draws and still be given
# the same assignment. Some retry a great deal before they are satisfied and
# need the roles held throughout; 31aa019c retries forever when it is, because
# the assignment is what it is failing on. Past this many rounds it is let go.
_ROUNDS = int(os.environ.get("SOLAR_PALETTE_ROUNDS", "30"))


def _caller_site():
    """The generator's own line, however many wrappers are stacked on choice.

    Counting frames breaks the moment a second wrapper goes on: the colour hold
    and the kind planner both patch `choice`, the planner is entered last so the
    generator calls it first, and when it declines and falls through to the
    colour hold, a fixed depth lands on the planner's wrapper in this file
    rather than on the line in generators.py. The colour test then looks for
    `cols` on the wrong line, finds nothing, and quietly stops holding --
    episodes sharing a colour fell from 0.99 to 0.88. Climbing out of this file
    is right whatever is stacked.
    """
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename.startswith(PIPELINE_DIR):
        f = f.f_back
    return (f.f_code.co_filename, f.f_lineno) if f is not None else None


def _asks_for_colour():
    """Is the call the generator is making a colour draw, by the name it passes?

    Values alone cannot tell: a colour pool is a subset of 0..9 and so is
    `interval(0, h, 1)` when the grid is ten cells high, and freezing that would
    pin every object to one position for the whole episode. The generators name
    what they pass -- `cols`, `remcols`, `ccols`, `colopts`, `itv` -- so the
    name is what decides.
    """
    key = _caller_site()
    if key is None:
        return False
    hit = _COLOUR_SITE.get(key)
    if hit is None:                      # one call site, one answer, decided once
        line = linecache.getline(*key)
        m = COLOUR_ARGS.search(line)
        hit = _COLOUR_SITE[key] = bool(m and COLOUR_NAME.search(m.group(1)))
    return hit


def _fix_colours(seq):
    """The pool as a list, when it is a pool of colours."""
    try:
        items = list(seq)
    except Exception:
        return None
    if not all(isinstance(x, int) and 0 <= x <= 9 for x in items):
        return None
    return items


class _HeldColours:
    """Patch generators.choice/sample for one episode, restore after."""

    def __init__(self, perm, genfn=None, roles=0):
        self.perm = perm
        self.genfn = genfn
        self.roles = roles

    def __enter__(self):
        # Every maker begins by deleting utils, dsl and generators from
        # sys.modules so its own embedded copy of the DSL wins, so the module
        # object is not a reliable handle -- the generator we hold was captured
        # before that and still reads its names out of the dictionary it was
        # defined in. Patch that dictionary.
        self.taken = 0
        self.rounds = 0
        self.first_pool = frozenset()
        self._g = self.genfn.__globals__
        self._choice = self._g.get("choice")
        self._sample = self._g.get("sample")
        perm = self.perm

        def choice(seq):
            items = _fix_colours(seq) if _asks_for_colour() else None
            if items is not None and self.taken >= self.roles \
                    and set(items) == self.first_pool and self.rounds < _ROUNDS:
                self.taken = 0        # the generator is retrying; roles again
                self.rounds += 1
            if items is not None and self.taken < self.roles:
                if self.taken == 0:
                    self.first_pool = set(items)
                # The role assignments come first, one per colour the maker
                # names, and everything after them is painting -- 0dfd9992 picks
                # a colour per cell of its pattern with the same call, and
                # holding those makes the pattern one flat colour, which the
                # verifier then refuses over and over. Counting the draws
                # separates them; the pool cannot, because 7b6016b9 takes both
                # of its colours out of the same one.
                pool = set(items)
                for c in perm[self.taken:]:
                    if c in pool:
                        self.taken += 1
                        return c
            return self._choice(seq)

        def sample(seq, k):
            items = _fix_colours(seq) if _asks_for_colour() else None
            if items is not None and self.taken >= self.roles \
                    and set(items) == self.first_pool and self.rounds < _ROUNDS:
                self.taken = 0
                self.rounds += 1
            if items is not None and self.taken < self.roles:
                pool = set(items)
                if self.taken == 0:
                    self.first_pool = pool
                picked = [c for c in perm[self.taken:] if c in pool][:k]
                if len(picked) == k:
                    self.taken += k
                    return picked
            return self._sample(seq, k)

        self._g["choice"], self._g["sample"] = choice, sample
        try:
            self.genfn._held = self
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        self._g["choice"], self._g["sample"] = self._choice, self._sample
        return False


def _maker_palette(gm_mod):
    """The colours this task's maker says are roles, in the order it names them.

    Which colours carry meaning is a per-task judgement and the maker already
    made it: sample_colors() picks one assignment for a whole episode, and its
    comments say why -- 0dfd9992 notes that the patch colour "has to stay the
    same across the episode for the test to be readable". Where colour is not a
    role the same function says so by returning nothing: 0d3d703e's reads "no
    colors are freely sampled ... only which subset of the 8 input colors
    appears varies (structural, not a role)", and holding it would flatten the
    variation the task is made of.

    So the answer to whether to hold, and to what, comes from here rather than
    from anything this file could work out on its own.
    """
    fn = getattr(gm_mod, "sample_colors", None)
    if fn is None:
        return None
    try:
        import inspect as _i
        got = fn(num_examples=3) if "num_examples" in _i.signature(fn).parameters else fn()
    except Exception:
        return None
    if not isinstance(got, dict) or not got:
        return None                      # the maker says colour is not a role here
    order = []
    for k, v in got.items():
        if k in ("category_plan", "instance_plan"):
            continue
        for c in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(c, int) and 0 <= c <= 9 and c not in order:
                order.append(c)
    if not order:
        return None
    # The count matters as much as the colours: it is how many draws at the top
    # of a generator are role assignments rather than painting.
    return order + [c for c in range(10) if c not in order], len(order)



def _is_case_pool(seq):
    """A small set of alternatives the generator picks the instance's kind from.

    `choice((True, False))` and `choice((identity, rot90, rot180, rot270))` are
    how re-arc decides whether this instance is transposed, or which way round
    it is -- the thing a maker's instance_plan exists to spread across an
    episode so the test asks about a kind the demonstrations showed. Left to
    chance, three examples land on one side often enough to matter: 2204b7a8
    misses on one episode in ten, d4469b4b -- which keeps three kinds behind a
    (colour, shape) table -- on five.
    """
    try:
        items = list(seq)
    except Exception:
        return None
    if not 2 <= len(items) <= 5:
        return None
    if all(isinstance(x, bool) for x in items):
        return items
    if all(callable(x) for x in items):
        return items
    # d4469b4b keeps its kinds as (colour, grid) pairs -- colabc = ((2, A),
    # (1, B), (3, C)), A, B and C being the three shapes those colours stand
    # for. That has to be told from a coordinate, which is also a short tuple:
    # 137eaa0f chooses a dot's location out of a handful of free cells, and
    # holding *that* to a walk over options pinned the dots and narrowed the
    # very thing its examples were meant to vary -- it went from missing on 2
    # episodes in 10 to 6. A coordinate is integers all the way down; a case
    # table has a grid in it.
    def _is_case_row(x):
        return (isinstance(x, tuple) and len(x) == 2
                and isinstance(x[0], int) and not isinstance(x[0], bool)
                and isinstance(x[1], tuple) and bool(x[1])
                and all(isinstance(r, tuple) for r in x[1]))

    if all(_is_case_row(x) for x in items):
        return items
    return None


class _PlannedCases:
    """Walk the kinds across an episode's pairs, and let the test repeat one.

    Applied where the choice is made rather than by picking among finished
    instances, so every pair is still exactly what re-arc's generator produced.
    Keyed by call site, so a generator making two such choices keeps them apart.
    """

    def __init__(self, genfn, n_examples):
        self.genfn = genfn
        self.n_examples = n_examples
        self.index = 0                    # which pair of the episode is being drawn
        self.seen = {}                    # call site -> the options it offered
        self.fired = set()                # sites already used for this instance
        self.dead = set()                 # sites reached twice: coins, not kinds

    def instance(self):
        """A new instance is starting; forget which sites have fired."""
        self.fired = set()



    def __enter__(self):
        self._g = self.genfn.__globals__
        self._choice = self._g.get("choice")
        outer = self

        def choice(seq):
            items = _is_case_pool(seq)
            if items is not None:
                key = _caller_site()
                if key is not None and key in outer.fired:
                    # A kind is settled once for the instance. A site reached
                    # twice in one draw is a coin inside a loop, and holding
                    # one is what stops the loop turning.
                    outer.dead.add(key)
                if key is not None and key not in outer.dead:
                    outer.fired.add(key)
                    opts = outer.seen.setdefault(key, items)
                    i = outer.index
                    if i < outer.n_examples:
                        return opts[i % len(opts)]
                    return opts[_random.randrange(min(outer.n_examples, len(opts)))]
            return outer._choice(seq)

        self._g["choice"] = choice
        return self

    def __exit__(self, *exc):
        self._g["choice"] = self._choice
        return False


ASSIGNED = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(?:randint|unifint)\(")


def _maker_plan(gm_mod):
    """The quantities this task's maker plans, one entry per pair.

    Some kinds are not chosen from a set but drawn as a number -- how many red
    blocks, how many colours -- and there is no `choice` to intercept. Which
    numbers those are cannot be read off their names: instrumenting randint and
    unifint and keeping every draw whose name reads as a count fires on 347 of
    400 generators, because re-arc draws a quantity of noise or objects almost
    everywhere and nearly all of them are difficulty knobs.

    What separates a kind from a knob is that the maker planned it. The plan
    arrives keyed by name -- nred, ncorns, numlins -- so the condition is that
    the plan names it and the generator has a draw assigned to the same name.
    That is eight tasks, and it is the same move as taking the colour roles
    from sample_colors instead of guessing them.
    """
    fn = getattr(gm_mod, "sample_colors", None)
    if fn is None:
        return None
    try:
        import inspect as _i
        got = fn(num_examples=3) if "num_examples" in _i.signature(fn).parameters else fn()
    except Exception:
        return None
    if not isinstance(got, dict):
        return None
    plan = got.get("instance_plan")
    if not plan or not all(isinstance(e, dict) for e in plan):
        return None
    keys = {k for e in plan for k, v in e.items() if isinstance(v, int)}
    return (plan, keys) if keys else None


class _PlannedCounts:
    """Hand the generator the quantity the maker planned for this pair.

    Only for names the plan itself mentions, so a difficulty knob the maker did
    not plan is left to the generator. The plan already states a value per pair
    and repeats one of them for the test, so nothing here invents a range or
    cycles a subset of one.
    """

    def __init__(self, genfn, plan, keys):
        self.genfn = genfn
        self.plan = plan
        self.keys = keys
        self.index = 0

    def _planned(self):
        if not self.plan:
            return None
        return self.plan[min(self.index, len(self.plan) - 1)]

    def __enter__(self):
        self._g = self.genfn.__globals__
        self._randint = self._g.get("randint")
        self._unifint = self._g.get("unifint")
        outer = self

        def named():
            site = _caller_site()
            if site is None:
                return None
            m = ASSIGNED.match(linecache.getline(*site))
            return m.group(1) if m else None

        def randint(a, b):
            n = named()
            if n in outer.keys:
                want = (outer._planned() or {}).get(n)
                if isinstance(want, int) and a <= want <= b:
                    return want
            return outer._randint(a, b)

        def unifint(lb, ub, rng):
            n = named()
            if n in outer.keys:
                want = (outer._planned() or {}).get(n)
                if isinstance(want, int) and rng[0] <= want <= rng[1]:
                    return want
            return outer._unifint(lb, ub, rng)

        if self._randint is not None:
            self._g["randint"] = randint
        if self._unifint is not None:
            self._g["unifint"] = unifint
        return self

    def __exit__(self, *exc):
        if self._randint is not None:
            self._g["randint"] = self._randint
        if self._unifint is not None:
            self._g["unifint"] = self._unifint
        return False

