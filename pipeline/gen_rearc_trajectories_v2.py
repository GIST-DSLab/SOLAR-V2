#!/usr/bin/env python3
"""
V2 trajectory generator: same as gen_rearc_trajectories.py but records the full
O2ARC object state at every step.

New fields per step (beyond v1):
  selected       : obs['selected']                   — env selection mask AFTER op
  object_active  : obs['object_states']['active']    — 1 if object mode active
  object         : obs['object_states']['object']    — extracted object pixels
  object_sel     : obs['object_states']['object_sel']— object shape mask
  object_dim     : obs['object_states']['object_dim']— (h, w) of object
  object_pos     : obs['object_states']['object_pos']— (r, c) top-left of object
  background     : obs['object_states']['background']— grid with object removed

This format supports Move/Rotate/Flip trajectories where the object position
changes between steps, so the model can track the object across operations.

Usage:
    export SOLAR_DATA_ROOT=<where the rollouts should land>
    python gen_rearc_trajectories_v2.py --subfolder arc-agi-1 \\
        --num_samples 100 --data_folder $SOLAR_DATA_ROOT/draw0
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

# The repo keeps this in pipeline/ and the working tree at its root, so a fixed
# parents[N] is wrong in one of them — copying the file between the two put
# MAKER_BASE outside the tree. Find the ancestor that actually holds maker/.
SOLAR_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "maker").is_dir())
sys.path.insert(0, str(SOLAR_ROOT))

import utils as solar_utils
# A maker set under maker/ can be a symlink to another disk. Each maker
# resolves its own root with .resolve(), which follows that link out of the
# repository, so re-arc never reaches sys.path from there — put it on here,
# where the root is known. It goes on AFTER the import above: re-arc has a
# utils.py of its own, and this line would otherwise shadow ours.
sys.path.insert(0, str(SOLAR_ROOT / "re-arc"))

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples",   type=int, default=10)
parser.add_argument("--num_examples",  type=int, default=3)
parser.add_argument("--rand_seed",     type=int, default=0)
parser.add_argument("--max_grid_dim",  nargs=2, type=int, default=[30, 30])
parser.add_argument("--force_grid_size", action="store_true", default=False,
                    help="Ask the maker to generate within --max_grid_dim instead of "
                         "discarding samples that overshoot it. Needs a maker built "
                         "from a gen_rearc_makers.py that supports max_grid_dim.")
parser.add_argument("--data_folder",   type=str,
                    default=str(Path(os.environ.get(
                        "SOLAR_DATA_ROOT", Path.cwd() / "solar-data")) / "rollout"),
                    help="output root; episodes land in <data_folder>/whole "
                         "(default: $SOLAR_DATA_ROOT/rollout)")
parser.add_argument("--subfolder",     type=str, default="arc-agi-1")
parser.add_argument("--skip_on_error", action="store_true", default=True)
parser.add_argument("--v1", action="store_true", default=False,
                    help="Output v1-compatible format (no object_states fields).")
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Only regenerate these task IDs (e.g. --tasks 045e512c 025d127b).")
parser.add_argument("--only_failures", action="store_true", default=False,
                    help="Save ONLY samples whose output is wrong (for inspecting failures).")
parser.add_argument("--demo_trajectories", action="store_true", default=False,
                    help="Also emit a full trajectory for every demonstration example, "
                         "not just the problem pair. Each maker-sample of N demos + 1 "
                         "problem is expanded into N+1 targets sharing a group_id, with "
                         "the N demonstrations fixed as the worked examples for every "
                         "target; each row is tagged role=problem|demo. Opt-in; off "
                         "leaves output byte-identical to before.")
parser.add_argument("--verify_filter", action="store_true", default=False,
                    help="re-arc-style augmentation gate: keep only generated instances "
                         "whose pairs pass the RE-ARC verifier verify_<task>; drop the "
                         "rest. Off by default (output byte-identical to before).")
parser.add_argument("--rearc_root", type=str, default=None,
                    help="Path to re-arc/ holding verifiers.py (default: SOLAR_ROOT/re-arc).")
parser.add_argument("--rearc_generate", action="store_true", default=False,
                    help="Augment with the ORIGINAL re-arc generate_<task> (verifier-matched "
                         "by construction) instead of the maker's own generate(); the only "
                         "adjustment is capping grids to --max_grid_dim. The maker still "
                         "supplies derive_operations for the trajectory.")
args = parser.parse_args()

MAX_GRID_DIM = tuple(args.max_grid_dim)
H, W = MAX_GRID_DIM
formatted_time = datetime.now().strftime("%y.%m.%d")

# ── verify-filter helpers (only touched when --verify_filter) ────────────────
_VERIFIERS = None
def _get_verifier(tid):
    """Return verify_<tid> from re-arc/verifiers.py, or None if unavailable."""
    global _VERIFIERS
    if _VERIFIERS is None:
        rr = args.rearc_root or str(SOLAR_ROOT / "re-arc")
        if rr not in sys.path:
            sys.path.insert(0, rr)
        try:
            import verifiers as _V
            _VERIFIERS = _V
        except Exception:
            _VERIFIERS = False
    if not _VERIFIERS:
        return None
    return getattr(_VERIFIERS, f"verify_{tid}", None)

def _pair_ok(vfn, gin, gout):
    """True if verify_<task>(gin) reproduces gout."""
    try:
        got = vfn(tuple(tuple(int(x) for x in row) for row in np.asarray(gin).tolist()))
        return [list(r) for r in got] == [list(r) for r in np.asarray(gout).tolist()]
    except Exception:
        return False

# ── original-re-arc-generate augmentation (only touched when --rearc_generate) ──
import linecache
import re
import time as _time
import random as _random
try:
    from arcle.loaders import Loader as _ArcleLoader
except Exception:
    _ArcleLoader = object

class _RearcLoader(_ArcleLoader):
    """Serve re-arc generate_<task> instances (with maker-derived ops) to ARCLE."""
    def __init__(self, samples):
        self.samples = samples
        self._pathlist = [""]
        self.data = samples
    def get_path(self, **k):
        return [""]
    def parse(self, **k):
        return self.samples

_GENERATORS = None
def _get_generator(tid):
    global _GENERATORS
    if _GENERATORS is None:
        rr = args.rearc_root or str(SOLAR_ROOT / "re-arc")
        if rr not in sys.path:
            sys.path.insert(0, rr)
        try:
            import generators as _G
            _GENERATORS = _G
        except Exception:
            _GENERATORS = False
    if not _GENERATORS:
        return None
    return getattr(_GENERATORS, f"generate_{tid}", None)

# Set to a _PlannedCases while one is active. The once-per-instance rule needs
# to know when a new draw starts, and the only place that knows is the loop that
# calls the generator -- wrapping genfn to announce it would add a frame, and
# both the colour hold and the planner read the call site by walking out of this
# file, so a frame added here is a call site misread there.
_DRAWING = None


def _gen_capped(genfn, lb, ub, max_hw, vfn=None, retries=80, deadline=None):
    """re-arc generate, resampled until the grid fits max_hw (the only size modification)
    and — when vfn is given (re-arc-style gate) — until verify reproduces it, per pair."""
    Hc, Wc = max_hw
    for _ in range(retries):
        # The clock has to be read here too: one call into a generator that
        # builds thirty-by-thirty grids is not fast, and eighty of them is
        # minutes -- checking only in the caller's loop bounds nothing.
        if deadline is not None and _time.monotonic() > deadline:
            return None
        held = getattr(genfn, "_held", None)
        if held is not None:
            held.taken = 0             # the roles are re-assigned per instance
            held.rounds = 0            # and a generator may retry a few times
        if _DRAWING is not None:
            _DRAWING.instance()
        try:
            d = genfn(lb, ub)
        except Exception:
            continue
        I = np.array(d["input"], int); O = np.array(d["output"], int)
        if max(I.shape[0], O.shape[0]) > Hc or max(I.shape[1], O.shape[1]) > Wc:
            continue
        if vfn is not None and not _pair_ok(vfn, I, O):
            continue
        return I, O
    return None



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
    while f is not None and f.f_code.co_filename == __file__:
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

def _episode_pool(genfn, need, max_hw, vfn, ufn, unify, perm=None, roles=0,
                  plan=None):
    """One episode's pairs, all drawn under the same colour assignment."""
    pool, tries = [], 0
    if perm is None:
        perm = _random.sample(range(10), 10)
    ctx = _HeldColours(perm, genfn, roles) if unify else None
    if ctx is not None:
        ctx.__enter__()
    # A held palette narrows what the generator can land on, and for a few tasks
    # it narrows it a long way -- 29ec7d0e spent minutes not finding a fourth
    # pair. Give holding a short budget and let the fallback take over: an
    # episode drawn in mixed colours beats an episode that never arrives.
    budget = need * (8 if unify else 40)
    retries = 20 if unify else 80
    # A count is not a bound on time: a generator that builds thirty-by-thirty
    # grids takes long enough that twelve thousand calls is minutes, and one
    # task holding the draw for minutes is worse than that task shipping in
    # mixed colours. 0e206a2e was six of them.
    slack = float(os.environ.get("SOLAR_PALETTE_SECONDS", "10"))
    # The same bound either way. Holding makes a generator's own retries
    # deterministic -- it gets the same colours back and produces the same
    # rejected instance -- so a long fallback does not rescue the episode, it
    # only delays the draw. Fifteen episodes are drawn and ten are needed.
    deadline = _time.monotonic() + slack
    # Only where the maker planned something. This is the one intervention that
    # fired on shape alone -- the colour hold asks the maker which colours are
    # roles, the count extension asks the plan which numbers it names, and this
    # took any `choice((True, False))` it saw. 72ca375d has no instance_plan at
    # all and hung a whole draw: its coin is flipped inside a `while True`,
    # holding it stopped the loop's exit condition from ever coming true, and
    # no retry budget helps because control never returns to one.
    cases = _PlannedCases(genfn, need - 1) if plan else None
    if cases is not None:
        cases.__enter__()
        globals()["_DRAWING"] = cases
    counts = _PlannedCounts(genfn, *plan) if plan else None
    if counts is not None:
        counts.__enter__()
    try:
        while len(pool) < need and tries < budget and _time.monotonic() < deadline:
            tries += 1
            if cases is not None:
                cases.index = len(pool)   # the examples first, then the test
            if counts is not None:
                counts.index = len(pool)
            lb = _random.random() * 0.8
            pr = _gen_capped(genfn, lb, min(1.0, lb + 0.3), max_hw,
                             vfn=vfn, retries=retries, deadline=deadline)
            if pr is not None:
                pool.append(pr)
    finally:
        if counts is not None:
            counts.__exit__()
        if cases is not None:
            cases.__exit__()
            globals()["_DRAWING"] = None
        if ctx is not None:
            ctx.__exit__()
    return pool if len(pool) >= need else None

def _build_rearc_samples(tid, gm_mod, n_samples, n_examples, max_hw):
    """Instances from re-arc generate_<task> (size-capped) + maker's derive_operations.
    With --verify_filter, every pair is resampled until verify reproduces it (re-arc style)."""
    genfn = _get_generator(tid)
    derive = getattr(gm_mod, "derive_operations", None)
    if genfn is None or derive is None:
        return None
    vfn = _get_verifier(tid) if args.verify_filter else None
    _random.seed(args.rand_seed)
    need = n_examples + 1
    samples = []
    # Unification needs the verifier whether or not --verify_filter asked for
    # one: it is what decides that a recolouring left the task alone.
    ufn = vfn if vfn is not None else _get_verifier(tid)
    # Decide once, not per episode. Holding narrows what the generator can land
    # on, and for a handful of tasks it narrows it far enough that filling a
    # pool takes minutes -- paying that fifteen times over, and then paying the
    # fallback fifteen times too, is what made the draw crawl. One probe says
    # whether this task can be drawn on a held palette at all.
    # One assignment per episode, not per task. sample_colors() is written to be
    # called once an episode -- that is how the maker path used it -- so calling
    # it once and reusing the answer gave all fifteen episodes of a task the
    # same colours. Within an episode the roles hold; between episodes they are
    # drawn again.
    got = _maker_palette(gm_mod)
    perm, roles = got if got else (None, 0)
    plan = _maker_plan(gm_mod)
    holds = perm is not None
    first = None
    if holds:
        first = _episode_pool(genfn, need, max_hw, vfn, ufn, unify=True,
                              perm=perm, roles=roles, plan=plan)
        if first is None:
            holds = False
    # A bound on the task as well as on each episode. 31aa019c can spend its ten
    # seconds an episode and still not fill a pool, and fifteen of those is two
    # and a half minutes for one task; past a minute the palette is given up and
    # the rest of the episodes are drawn plainly.
    task_deadline = _time.monotonic() + 60.0
    for s in range(n_samples):
        if holds and _time.monotonic() > task_deadline:
            holds = False
        if first is not None:
            pool, first = first, None
        else:
            fresh = _maker_palette(gm_mod) if holds else None
            pool = _episode_pool(genfn, need, max_hw, vfn, ufn, unify=holds,
                                 perm=fresh[0] if fresh else None,
                                 roles=fresh[1] if fresh else 0,
                                 plan=_maker_plan(gm_mod))
        if pool is None:
            continue
        ex, (I, O) = pool[:n_examples], pool[n_examples]
        try:
            ops, sels = derive(I.tolist(), O.tolist())
        except Exception:
            continue
        ei = [p[0].astype(np.uint8) for p in ex]
        eo = [p[1].astype(np.uint8) for p in ex]
        desc = {"operations": ops, "selections": sels, "id": str(s)}
        samples.append((ei, eo, [I.astype(np.uint8)], [O.astype(np.uint8)], desc))
    return samples

MAKER_BASE = SOLAR_ROOT / "maker" / args.subfolder
task_dirs  = sorted(MAKER_BASE.iterdir())
if args.tasks:
    task_dirs = [d for d in task_dirs if d.name in args.tasks]

whole_root = Path(args.data_folder) / "whole"
whole_root.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pad(arr: np.ndarray, fill: int = 10) -> np.ndarray:
    """Pad any (H', W') ndarray to MAX_GRID_DIM with fill value."""
    out = np.full(MAX_GRID_DIM, fill, dtype=np.int8)
    h, w = min(arr.shape[0], H), min(arr.shape[1], W)
    out[:h, :w] = arr[:h, :w]
    return out


def _pad_bool(arr: np.ndarray) -> np.ndarray:
    """Pad a boolean/uint8 selection mask to MAX_GRID_DIM with zeros."""
    out = np.zeros(MAX_GRID_DIM, dtype=np.int8)
    h, w = min(arr.shape[0], H), min(arr.shape[1], W)
    out[:h, :w] = arr[:h, :w]
    return out


def _record_step(data: dict, obs: dict, action_sel_bbox, action_op: int,
                 reward: int, term: bool, step: int, v2: bool = True) -> None:
    """Append one step worth of state to data dict."""
    grid_template = np.full(MAX_GRID_DIM, 10, dtype=np.int8)

    gd_h, gd_w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
    c_h,  c_w  = int(obs["clip_dim"][0]), int(obs["clip_dim"][1])

    grid_pad = grid_template.copy()
    grid_pad[:gd_h, :gd_w] = obs["grid"][:gd_h, :gd_w]

    clip_pad = grid_template.copy()
    if c_h > 0 and c_w > 0:
        clip_pad[:c_h, :c_w] = obs["clip"][:c_h, :c_w].astype(np.int8)

    data["grid"].append(grid_pad.tolist())
    data["grid_dim"].append([gd_h, gd_w])
    data["clip"].append(clip_pad.tolist())
    data["clip_dim"].append([int(c_h), int(c_w)])
    _sel_mask = solar_utils.to_sel_mask(action_sel_bbox, MAX_GRID_DIM)
    data["selection"].append(solar_utils.mask_to_bbox(_sel_mask))
    data["selection_mask"].append(_sel_mask.tolist())
    data["operation"].append(action_op)
    data["operation_name"].append(solar_utils.mapping_operation(action_op))
    data["reward"].append(reward)
    data["terminated"].append(term)
    data["step"].append(step)

    if not v2:
        return

    # v2 fields — object state from env obs
    obj = obs["object_states"]
    data["selected"].append(_pad_bool(obs["selected"]).tolist())
    data["object_active"].append(int(obj["active"]))
    data["object"].append(_pad(obj["object"], fill=0).tolist())
    data["object_sel"].append(_pad_bool(obj["object_sel"]).tolist())
    data["object_dim"].append([int(obj["object_dim"][0]), int(obj["object_dim"][1])])
    data["object_pos"].append([int(obj["object_pos"][0]), int(obj["object_pos"][1])])
    data["background"].append(_pad(obj["background"], fill=0).tolist())


def _expand_demo_targets(loader, gm_mod) -> None:
    """Fixed-demonstration expansion for --demo_trajectories.

    Each maker-sample of N demos + 1 problem is rewritten into N+1 loader.data
    entries — every pair takes a turn as the "problem" (the reset target ARCLE
    replays), while the worked examples stay FIXED to the sample's N
    demonstrations for every target (the test pair is never shown as a
    demonstration). The problem reuses its precomputed ops; each demo derives
    fresh ops via the maker's own
    `derive_operations`. group_id/role/example_index ride in `desc`. The env is
    built AFTER this, so prob_index addresses every target. A demo whose derive
    raises is dropped here; one whose replay later mismatches is dropped by the
    existing validation in run_task — neither drops its siblings.
    """
    import inspect

    derive = gm_mod.derive_operations
    n_params = len(inspect.signature(derive).parameters)

    new_data = []
    for sample in loader.data:
        ex_in_list, ex_out_list, pr_in_list, pr_out_list, desc = sample
        if not pr_in_list or not pr_out_list:
            new_data.append(sample)          # degenerate sample: leave untouched
            continue

        base_id = str(desc["id"])
        concept = desc.get("concept", "")
        pairs = list(zip(ex_in_list, ex_out_list)) + [(pr_in_list[0], pr_out_list[0])]
        prob_idx = len(pairs) - 1

        for k, (Ik, Ok) in enumerate(pairs):
            if k == prob_idx:
                ops, sels = desc["operations"], desc["selections"]
                row_id, role = base_id, "problem"
            else:
                try:
                    ops, sels = derive(Ik, Ok) if n_params >= 2 else derive(Ik)
                except Exception:
                    continue                 # drop this demo target, keep siblings
                row_id, role = f"{base_id}_ex{k}", "demo"

            # Worked examples are the sample's FIXED demonstrations — the same
            # set for every target in the family, so the demonstration panel is
            # identical across all N+1 episodes. The problem (test) pair is never
            # shown as a demonstration; a demo target still sees the full demo
            # set (itself included) rather than a leave-one-out subset.
            ex_i = list(ex_in_list)
            ex_o = list(ex_out_list)
            new_data.append((ex_i, ex_o, [Ik], [Ok], {
                "id":            row_id,
                "concept":       concept,
                "operations":    ops,
                "selections":    sels,
                "group_id":      base_id,
                "role":          role,
                "example_index": k,
            }))

    loader.data = new_data


def run_task(tid: str, maker_path: Path) -> tuple[int, int]:
    """Generate trajectories for one task. Returns (correct, total)."""

    spec = importlib.util.spec_from_file_location(f"gm_{tid}", str(maker_path))
    gm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm_mod)
    GridMaker = gm_mod.GridMaker

    if args.rearc_generate:
        samples = _build_rearc_samples(tid, gm_mod, args.num_samples,
                                       args.num_examples, (H, W))
        if not samples:
            tqdm.write(f"  SKIP {tid}: --rearc_generate produced no valid samples")
            return 0, 0
        loader = _RearcLoader(samples)
    else:
        loader = GridMaker(
            rand_seed=args.rand_seed,
            num_samples=args.num_samples,
            num_examples=args.num_examples,
            # Makers that understand it generate within the ceiling instead of
            # letting the oversize filter below throw the sample away. Omit the
            # kwarg entirely when the flag is off — passing None instead makes every
            # maker that unpacks it into (max_h, max_w) raise on the first sample.
            **({"max_grid_dim": MAX_GRID_DIM} if args.force_grid_size else {}),
        )

    # Expand demos into their own targets BEFORE the env is built, so prob_index
    # addresses every one of them.
    if args.demo_trajectories:
        _expand_demo_targets(loader, gm_mod)

    env = gym.make(
        "ARCLE/O2ARCv2Env-v0",
        render_mode=None,
        data_loader=loader,
        max_grid_size=MAX_GRID_DIM,
        colors=10,
        max_episode_steps=None,
        max_trial=1,
    )

    folder = whole_root / f"test.{tid}.s{H}.{formatted_time}"
    folder.mkdir(exist_ok=True)

    correct = total = 0
    dropped = 0
    vfn = _get_verifier(tid) if args.verify_filter else None
    if args.verify_filter and vfn is None:
        tqdm.write(f"  NOTE {tid}: --verify_filter on but verify_{tid} not found; not filtering")

    for i, sample in enumerate(loader.data):
        ex_in_list, ex_out_list, pr_in_list, pr_out_list, desc = sample
        if not pr_in_list or not pr_out_list:
            continue

        # re-arc-style augmentation gate: only keep instances the verifier reproduces
        # (problem pair AND every worked example). Drop the rest.
        if vfn is not None:
            ok = _pair_ok(vfn, pr_in_list[0], pr_out_list[0]) and all(
                _pair_ok(vfn, ei, eo) for ei, eo in zip(ex_in_list, ex_out_list))
            if not ok:
                dropped += 1
                continue

        ops  = desc["operations"]
        sels = desc["selections"]
        pr_out = pr_out_list[0]
        g_h, g_w = pr_out.shape

        # skip samples whose grid exceeds MAX_GRID_DIM
        if g_h > H or g_w > W:
            continue

        try:
            obs, info = env.reset(options={"prob_index": i, "adaptation": False})
        except Exception as e:
            tqdm.write(f"  SKIP {tid}[{i}] reset: {e}")
            continue

        grid_template = np.full(MAX_GRID_DIM, 10, dtype=np.int8)

        _desc_out = {"id": desc["id"], "concept": desc.get("concept", "")}
        # Provenance for --demo_trajectories rides in desc so the round-trip
        # exporter (which ignores desc) needs no special-casing. Absent when the
        # flag is off, leaving the record byte-identical to before.
        for _extra in ("group_id", "role", "example_index"):
            if _extra in desc:
                _desc_out[_extra] = desc[_extra]

        data = {
            "desc":         _desc_out,
            "step":         [],
            "selection":    [],
            "operation":    [],
            "operation_name": [],
            "reward":       [],
            "terminated":   [],
            "grid_dim":     [],
            "grid":         [],
            "clip_dim":     [],
            "clip":         [],
            "selection_mask": [],
            "in_grid":      None,
            "out_grid":     None,
            "ex_in":        [],
            "ex_out":       [],
            "ex_in_grid_dim":  [],
            "ex_out_grid_dim": [],
        }
        if not args.v1:
            data.update({
                "selected":      [],
                "object_active": [],
                "object":        [],
                "object_sel":    [],
                "object_dim":    [],
                "object_pos":    [],
                "background":    [],
            })

        v2 = not args.v1

        # step 0: initial state (no action yet)
        _record_step(data, obs, sels[0], ops[0], 0, False, 0, v2=v2)

        # step 1..N: execute each op
        step_err = False
        for s, (op, sel) in enumerate(zip(ops, sels)):
            sel_mask = solar_utils.to_sel_mask(sel, MAX_GRID_DIM)
            action   = {"selection": sel_mask.astype(bool), "operation": op}
            try:
                obs, reward, term, trunc, info = env.step(action)
            except Exception as e:
                tqdm.write(f"  SKIP {tid}[{i}] step {s} op={op}: {e}")
                step_err = True
                break

            next_sel = sels[s + 1] if s + 1 < len(sels) else sel
            next_op  = ops[s + 1]  if s + 1 < len(ops)  else op
            _record_step(data, obs, next_sel, next_op, int(reward), bool(term), s + 1, v2=v2)

        if step_err:
            continue

        # Drop the terminal filler action: the last recorded step is the post-Submit
        # terminal STATE (it keeps its grid, reward and terminated=True), but it has no
        # action to take from it. Recording one made every trajectory end in Submit,Submit.
        # Result: operation/operation_name have length N (the real actions, ending in a
        # single Submit) while the state arrays stay N+1 — the standard RL alignment.
        data["operation"] = data["operation"][:-1]
        data["operation_name"] = data["operation_name"][:-1]

        data["in_grid"]  = data["grid"][0]
        data["out_grid"] = data["grid"][-1]

        for ei, eo in zip(ex_in_list, ex_out_list):
            solar_utils.append_example(data, ei, eo, grid_template.copy())

        # validate
        final = np.array(data["grid"][-1])[:g_h, :g_w]
        ok = np.array_equal(final.astype(np.uint8), pr_out)
        total += 1
        correct += ok

        if not ok:
            tqdm.write(f"  FAIL {tid}[{i}]: ops={ops[:4]}")
            if not args.only_failures:
                continue
        elif args.only_failures:
            continue                       # only_failures mode: keep just the wrong ones

        data = solar_utils.convert_npint_to_int(data)
        # The filename comes from the maker's own id, and nothing obliges a maker
        # to vary it per sample: the handcraft set returns a constant, so 25
        # samples all wrote one path and 24 were silently lost. Only the last
        # survived, and the run still reported 375/375 correct. Disambiguate on
        # collision rather than trusting the id to be unique.
        stem = str(desc["id"])
        path = folder / f"{stem}.json"
        if path.exists():
            k = 1
            while (folder / f"{stem}_{k}.json").exists():
                k += 1
            path = folder / f"{stem}_{k}.json"
        with open(path, "w") as f:
            json.dump(data, f)

    if vfn is not None and dropped:
        tqdm.write(f"  {tid}: verify_filter dropped {dropped} instance(s); kept {total}")
    return correct, total


# ── main loop ─────────────────────────────────────────────────────────────────
total_correct = total_total = 0
failed_tasks = []

for task_dir in tqdm(task_dirs, desc="tasks"):
    maker_path = task_dir / "grid_maker.py"
    if not maker_path.exists():
        continue
    tid = task_dir.name
    try:
        c, t = run_task(tid, maker_path)
        total_correct += c
        total_total   += t
        if c < t:
            failed_tasks.append((tid, c, t))
    except Exception as e:
        failed_tasks.append((tid, "ERR", str(e)[:80]))
        if not args.skip_on_error:
            raise
        tqdm.write(f"ERROR {tid}: {e}")

print(f"\nDone: {total_correct}/{total_total} correct "
      f"({100*total_correct/max(total_total,1):.1f}%)")
print(f"Output: {whole_root}")
if failed_tasks:
    print(f"Failed ({len(failed_tasks)}):")
    for f in failed_tasks:
        print(" ", f)
