"""
ARC Task: ecdecbb3 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors / plan

# The only discrete structural variant is the final dmirror: the guide lines are
# either horizontal rows or vertical columns.  Cover both across the examples.
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # dotc must be non-zero: the dot is grabbed and slid with ARCLE Move ops,
    # and ARCLE's object buffer only holds non-zero cells.
    dotc = random.choice(list(range(1, 10)))
    rest = [c for c in range(10) if c != dotc]
    bgc, linc = random.sample(rest, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


# ---------------------------------------------------------------- generator

def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, dotc=None, linc=None, transpose=None) -> dict:
    if transpose is None:
        transpose = choice((True, False))
    if bgc is None or dotc is None or linc is None:
        kw = sample_colors()
        bgc, dotc, linc = kw["bgc"], kw["dotc"], kw["linc"]

    mh = max(4, min(30, int(max_h)))
    mw = max(4, min(30, int(max_w)))

    h = unifint(diff_lb, diff_ub, (4, mh))
    w = unifint(diff_lb, diff_ub, (4, mw))

    gi = canvas(bgc, (h, w))
    nl = unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    inds = interval(0, h, 1)
    locs = []
    for k in range(nl):
        if len(inds) == 0:
            break
        idx = choice(inds)
        locs.append(idx)
        inds = remove(idx, inds)
        inds = remove(idx - 1, inds)
        inds = remove(idx + 1, inds)
        inds = remove(idx - 2, inds)
        inds = remove(idx + 2, inds)
    locs = sorted(locs)
    for ii in locs:
        gi = fill(gi, linc, hfrontier((ii, 0)))

    iopts = difference(
        difference(difference(interval(0, h, 1), locs), apply(increment, locs)),
        apply(decrement, locs)
    )
    jopts = interval(0, w, 1)
    ndots = unifint(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    dlocs = []
    for k in range(ndots):
        if len(iopts) == 0 or len(jopts) == 0:
            break
        loci = choice(iopts)
        locj = choice(jopts)
        dlocs.append((loci, locj))
        jopts = remove(locj, jopts)
        jopts = remove(locj + 1, jopts)
        jopts = remove(locj - 1, jopts)

    go = gi
    for d in dlocs:
        loci, locj = d
        if loci < min(locs):
            go = fill(go, dotc, connect(d, (min(locs), locj)))
            go = fill(go, linc, neighbors((min(locs), locj)))
        elif loci > max(locs):
            go = fill(go, dotc, connect(d, (max(locs), locj)))
            go = fill(go, linc, neighbors((max(locs), locj)))
        else:
            sp = [e for e in locs if e < loci][-1]
            ep = [e for e in locs if e > loci][0]
            go = fill(go, dotc, connect((sp, locj), (ep, locj)))
            go = fill(go, linc, neighbors((sp, locj)))
            go = fill(go, linc, neighbors((ep, locj)))
        gi = fill(gi, dotc, {d})

    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- trajectory

def derive_operations(I, O):
    """
    Rule: every dot gravitates onto the nearest guide line; it leaves a trail of
    its own colour behind it (spanning the whole gap when it sits between two
    lines), and the line grows a 3x3 collar of line-colour around the contact.

    Route per dot:  unit Moves that carry the dot onto its line  ->  one Color
    that paints the trail it left  ->  one Color per contact for the collar.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # colour roles: bgc > linc > dotc by cell count (lines <= h*w/4, dots <= w/2)
    cnt = Counter(I.flatten().tolist())
    order = [c for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    bgc = order[0]
    dotc = order[-1]
    mids = [c for c in order if c != bgc and c != dotc]
    linc = mids[0] if mids else order[min(1, len(order) - 1)]

    horizontal = any(bool(np.all(I[r] == linc)) for r in range(h))

    if horizontal:
        locs = [r for r in range(h) if bool(np.all(I[r] == linc))]

        def pos(a, b):
            return (a, b)

        mop_pos, mop_neg = 21, 20            # a grows downward
    else:
        locs = [c for c in range(w) if bool(np.all(I[:, c] == linc))]

        def pos(a, b):
            return (b, a)

        mop_pos, mop_neg = 22, 23            # a grows rightward

    ops, sels = [], []

    if not locs:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    dots = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == dotc:
                dots.append((r, c) if horizontal else (c, r))   # (a, b)
    dots.sort()

    G = I.copy()

    for (a, b) in dots:
        if a < locs[0]:
            contacts = [locs[0]]
            lo, hi = a, locs[0]
        elif a > locs[-1]:
            contacts = [locs[-1]]
            lo, hi = locs[-1], a
        else:
            sp = max(l for l in locs if l < a)
            ep = min(l for l in locs if l > a)
            contacts = [sp, ep]
            lo, hi = sp, ep

        # --- gravitate: slide the dot onto the nearest line ------------------
        target = min(contacts, key=lambda l: (abs(l - a), l))
        n = abs(target - a)
        start = pos(a, b)
        dest = pos(target, b)
        if n > 0:
            mop = mop_pos if target > a else mop_neg
            ops.append(mop)
            sels.append(sel_of([start]))          # first Move GRABS the dot
            for _ in range(n - 1):
                ops.append(mop)
                sels.append(sel_of([]))           # empty -> keep sliding it
            G[start] = 0                          # ARCLE zeroes the footprint
            G[dest] = dotc

        # --- the trail the dot left behind -----------------------------------
        trail = [pos(x, b) for x in range(lo, hi + 1)]
        paint = [p for p in trail if G[p] != dotc]
        if paint:
            ops.append(int(dotc))
            sels.append(sel_of(paint))
            for p in paint:
                G[p] = dotc

        # --- the collar the line grows at each contact -----------------------
        for la in sorted(contacts, key=lambda l: abs(l - a)):
            cr, cc = pos(la, b)
            nb = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, ccc = cr + dr, cc + dc
                    if 0 <= rr < h and 0 <= ccc < w and G[rr, ccc] != linc:
                        nb.append((rr, ccc))
            if nb:
                ops.append(int(linc))
                sels.append(sel_of(nb))
                for p in nb:
                    G[p] = linc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox == the whole grid, intentionally
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
                # backwards-compatible single-key form; new makers use kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
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
                                f"for task ecdecbb3"
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
                    f"Failed to build a complete episode for task ecdecbb3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ecdecbb3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
