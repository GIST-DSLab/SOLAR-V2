"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# ----------------------------------------------------------------------------
# Task 3ac3eb23
#
# Input : a canvas of bgc carrying a few single marker cells, all of them on ONE
#         edge line of the grid (the generator puts them on the top row, then
#         rotates the whole grid by 0 / 90 / 180 / 270 degrees).
# Output: every marker grows a zig-zag fan straight into the grid: at EVEN depth
#         d the marker's colour sits on the marker's own lane, at ODD depth it
#         sits on the two lanes beside it.  The fan repeats with period 2 in
#         depth and runs the whole way across the grid.
#
# derive_operations reads all of that off I alone — O is never consulted — and
# then picks HOW to draw it from what it read: the end of the edge the nearest
# marker sits on decides the sweep direction, and the markers together with the
# strip length decide whether a fan is drawn as three dotted lanes, as one
# 2-deep unit stamped down the strip, or grown outward depth by depth.  A marker
# colour that is already standing somewhere is cloned from that fan.  Same input
# -> same route; the variety comes from the instances differing.
# ----------------------------------------------------------------------------

MF_VARIANTS = [{"mf": "identity"}, {"mf": "rot90"}, {"mf": "rot180"}, {"mf": "rot270"}]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    return random.randint(lo, hi)


def _rot90(g):        # CW
    return tuple(tuple(r) for r in zip(*g[::-1]))


def _rot270(g):       # CCW
    return tuple(tuple(r) for r in zip(*g))[::-1]


def _rot180(g):
    return tuple(tuple(r[::-1]) for r in g[::-1])


# ---------------------------------------------------------------- colors

def sample_colors(num_examples=None) -> dict:
    """bgc is the one colour role the rule needs pinned; the marker colours are
    free (a fan simply keeps its own marker's colour, 0 included).  The rotation
    — i.e. which edge the markers sit on — is the discrete structural variant,
    so it is planned per instance and every case is shown before the test."""
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(MF_VARIANTS):
        examples = [dict(v) for v in MF_VARIANTS]
        examples += [dict(random.choice(MF_VARIANTS)) for _ in range(n_ex - len(MF_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(MF_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------- generator

def generate(diff_lb, diff_ub, max_h, max_w, bgc, mf=None) -> dict:
    if mf is None:
        mf = random.choice([v["mf"] for v in MF_VARIANTS])

    # rot90 / rot270 swap the dimensions -> sample within the swapped caps
    if mf in ("rot90", "rot270"):
        cap_h, cap_w = max_w, max_h
    else:
        cap_h, cap_w = max_h, max_w
    cap_h = max(3, cap_h)
    cap_w = max(3, cap_w)

    h = _unifint(diff_lb, diff_ub, (3, cap_h))
    w = _unifint(diff_lb, diff_ub, (3, cap_w))

    remcols = [c for c in range(10) if c != bgc]
    nlocs = _unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))

    locopts = list(range(1, w - 1))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        banned = set(range(locj - 2, locj + 3))
        locopts = [x for x in locopts if x not in banned]
        col = random.choice(remcols)
        gi[0][locj] = col
        for p in range(0, h, 2):
            go[p][locj] = col
        for p in range(1, h, 2):
            go[p][locj - 1] = col
            go[p][locj + 1] = col

    gi = tuple(tuple(r) for r in gi)
    go = tuple(tuple(r) for r in go)

    if mf == "rot90":
        gi, go = _rot90(gi), _rot90(go)
    elif mf == "rot180":
        gi, go = _rot180(gi), _rot180(go)
    elif mf == "rot270":
        gi, go = _rot270(gi), _rot270(go)

    return {"input": gi, "output": go}


# ---------------------------------------------------------------- ops

def derive_operations(I, O):
    """Draw every marker's fan.  The edge the markers sit on, the inward
    direction, the length of each strip, the order the markers are taken in and
    the technique each fan is drawn with are all measured from I; O is never
    read, so a different answer grid returns the very same operations."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit = [0, 0, hi - 1, wi - 1]

    # the canvas colour: the markers are a handful of cells on an empty canvas
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    if not seeds:
        return [34], [submit]

    # the markers sit on exactly one edge line, never on a corner:
    #   u = inward (depth) direction, v = along the edge,
    #   D = how deep a strip runs, N = the length of the edge line itself
    if all(r == 0 for r, _ in seeds):
        u, v, D, N = (1, 0), (0, 1), hi, wi
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D, N = (-1, 0), (0, 1), hi, wi
    elif all(c == 0 for _, c in seeds):
        u, v, D, N = (0, 1), (1, 0), wi, hi
    else:
        u, v, D, N = (0, -1), (1, 0), wi, hi

    def cell(s, d, k):
        """the cell d deep into the grid and k lanes sideways from marker s"""
        return (s[0] + u[0] * d + v[0] * k, s[1] + u[1] * d + v[1] * k)

    def along(s):
        """position of s along the edge line"""
        return s[0] * v[0] + s[1] * v[1]

    def bbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), min(cs), max(rs), max(cs)

    # ---- route choice 1: which end of the edge to start from ----------------
    # begin at the end of the edge a marker sits closest to, and sweep across.
    ts = sorted(along(s) for s in seeds)
    ascending = ts[0] <= (N - 1 - ts[-1])
    order = sorted(seeds, key=along, reverse=not ascending)
    near_k = -1 if ascending else 1      # the wing on the side we came from
    far_k = -near_k                      # the wing on the side we head towards

    # ---- route choice 2: how a fan gets drawn -------------------------------
    # 0 = three dotted lanes, 1 = stamp one 2-deep unit down the strip,
    # 2 = grow the fan outward depth by depth (only while that stays small).
    style = (sum(int(I[r, c]) for r, c in seeds) + sum(ts) + len(seeds) + D) % 3
    if style == 2 and (D - 1) * len(seeds) > 30:
        style = 0                        # far too much to place cell by cell

    drawn = {}                           # colour -> a fan of it already standing
    for s in order:
        col = int(I[s[0], s[1]])
        P = D // 2                       # whole 2-deep units that fit in a strip

        if col != 0 and style != 2 and col in drawn:
            # this colour already stands somewhere: carry a copy of that whole
            # fan across to this marker's lane.  (Skipped for colour 0, which
            # Copy/Paste treats as nothing and would not carry.)
            prev = drawn[col]
            dt = along(s) - along(prev)
            r0, c0, r1, c1 = bbox([cell(prev, d, k)
                                   for d in (0, D - 1) for k in (-1, 1)])
            ops.append(29)
            sels.append([r0, c0, r1 - r0, c1 - c0])
            ops.append(30)
            sels.append([r0 + v[0] * dt, c0 + v[1] * dt, 0, 0])

        elif style == 1 and col != 0 and P >= 2:
            # complete the first 2-deep unit — the marker plus its two wings —
            # then stamp that unit on down the strip.  Colour 0 cannot ride in
            # the clipboard, so it never takes this route.
            ops.append(col)
            sels.append(sel_of([cell(s, 1, near_k), cell(s, 1, far_k)]))
            r0, c0, r1, c1 = bbox([cell(s, 0, 0), cell(s, 1, -1), cell(s, 1, 1)])
            ops.append(29)
            sels.append([r0, c0, r1 - r0, c1 - c0])
            for j in range(1, P):
                ops.append(30)
                sels.append([r0 + u[0] * 2 * j, c0 + u[1] * 2 * j, 0, 0])
            if D % 2 == 1:               # one lone even depth left at the far end
                ops.append(col)
                sels.append(sel_of([cell(s, D - 1, 0)]))

        elif style == 2:
            # grow the fan one step at a time, away from the marker
            for d in range(1, D):
                step = [cell(s, d, 0)] if d % 2 == 0 else \
                       [cell(s, d, near_k), cell(s, d, far_k)]
                ops.append(col)
                sels.append(sel_of(step))

        else:
            # three dotted lanes: the marker's own lane (its first cell is the
            # marker itself, which already holds the colour) and the two beside it
            centre = [cell(s, d, 0) for d in range(2, D, 2)]
            wings = {k: [cell(s, d, k) for d in range(1, D, 2)] for k in (-1, 1)}
            lanes = [centre, wings[near_k], wings[far_k]]
            if col % 2 == 1:             # the wings first, then the centre lane
                lanes = [wings[near_k], wings[far_k], centre]
            for lane in lanes:
                if lane:                 # nothing to lay down -> no operation
                    ops.append(col)
                    sels.append(sel_of(lane))

        drawn.setdefault(col, s)

    ops.append(34)
    sels.append(submit)
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task 3ac3eb23"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
