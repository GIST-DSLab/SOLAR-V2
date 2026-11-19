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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# The golf entry and the maker read the task the same way: a dot shoots a ray toward
# the lines, the line cell it hits takes the dot's colour, and the line wraps a 3x3
# collar around that hit. That part is unchanged here.
# They differ in what one action covers. The golf entry rotates the grid four times
# and grows each dot in one direction per pass, so each side of a dot is its own ray.
# The maker paints a dot sitting between two lines as a single line-to-line stroke.
# This version follows the golf reading:
#   for each dot -> for each side that has a line: one ray op, then that hit's collar.
# A dot between two lines therefore makes two rays and two collars, in that order.

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, linc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, mirror=None, **kw) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + diff_lb * (hi - lo)
        b = lo + diff_ub * (hi - lo)
        return int(round(random.uniform(a, b)))

    if mirror is None:
        mirror = random.choice([True, False])
    HM, WM = (max_w, max_h) if mirror else (max_h, max_w)
    h = unifint(4, min(30, HM))
    w = unifint(4, min(30, WM))
    gi = np.full((h, w), bgc, dtype=int)
    nl = unifint(1, h // 4)
    inds = list(range(h))
    locs = []
    for _ in range(nl):
        if not inds:
            break
        idx = random.choice(inds)
        locs.append(idx)
        inds = [x for x in inds if abs(x - idx) > 2]
    locs = sorted(locs)
    for ii in locs:
        gi[ii, :] = linc
    bad = set(locs) | {l + 1 for l in locs} | {l - 1 for l in locs}
    iopts = [i for i in range(h) if i not in bad]
    jopts = list(range(w))
    ndots = unifint(1, max(1, min(len(iopts), w // 2)))
    dlocs = []
    for _ in range(ndots):
        if not iopts or not jopts:
            break
        li = random.choice(iopts)
        lj = random.choice(jopts)
        dlocs.append((li, lj))
        jopts = [j for j in jopts if abs(j - lj) > 1]
    go = gi.copy()

    def nb(r, c):
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) != (r, c) and 0 <= rr < h and 0 <= cc < w:
                    go[rr, cc] = linc

    for li, lj in dlocs:
        if li < min(locs):
            go[li:min(locs) + 1, lj] = dotc
            nb(min(locs), lj)
        elif li > max(locs):
            go[max(locs):li + 1, lj] = dotc
            nb(max(locs), lj)
        else:
            sp = [e for e in locs if e < li][-1]
            ep = [e for e in locs if e > li][0]
            go[sp:ep + 1, lj] = dotc
            nb(sp, lj)
            nb(ep, lj)
        gi[li, lj] = dotc
    if mirror:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Lines: full rows (or full columns) of one single non-background colour.
    row_lines = [r for r in range(hi) if I[r, 0] != bgc and len(set(I[r].tolist())) == 1]
    horiz = len(row_lines) > 0
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(r for r in range(h) if T[r, 0] != bgc and len(set(T[r].tolist())) == 1)
    linc = int(T[locs[0], 0])
    dotc = int([c for c in set(T.flatten().tolist()) if c not in (bgc, linc)][0])

    def mp(r, c):
        return (r, c) if horiz else (c, r)

    G = T.copy()
    dots = sorted((r, c) for r in range(h) for c in range(w)
                  if T[r, c] == dotc and r not in locs)

    # Each dot fires one ray toward each side that has a line. The ray stops ON the
    # nearest line on that side, the line cell turns dot-coloured, and the line wraps
    # a 3x3 collar around the hit.
    for dr, dc in dots:
        targets = []
        above = [l for l in locs if l < dr]
        below = [l for l in locs if l > dr]
        if above:
            targets.append(max(above))
        if below:
            targets.append(min(below))
        for t in targets:
            step = -1 if t < dr else 1
            ray = [(r, dc) for r in range(dr + step, t + step, step) if G[r, dc] != dotc]
            if ray:
                ops.append(dotc)
                sels.append(sel_of([mp(r, c) for r, c in ray]))
                for r, c in ray:
                    G[r, c] = dotc
            collar = [(r, c) for r in range(t - 1, t + 2) for c in range(dc - 1, dc + 2)
                      if 0 <= r < h and 0 <= c < w and (r, c) != (t, dc) and G[r, c] != linc]
            if collar:
                ops.append(linc)
                sels.append(sel_of([mp(r, c) for r, c in collar]))
                for r, c in collar:
                    G[r, c] = linc

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
