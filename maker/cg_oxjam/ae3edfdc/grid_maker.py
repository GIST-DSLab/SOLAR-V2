"""
ARC Task: ae3edfdc (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    # Colours 1, 2, 3 and 7 are fixed by the generator. Only the background is random.
    cols = [c for c in range(10) if c not in (1, 2, 3, 7)]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    from dsl import (canvas, asindices, neighbors, sfilter, totuple, fill,
                     position, add)
    from utils import unifint
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 7)])
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    rdi = random.randint(1, h - 2)
    rdj = random.randint(1, w - 2)
    rd = (rdi, rdj)
    reminds = inds - ({rd} | neighbors(rd))
    reminds = sfilter(reminds, lambda ij: 1 <= ij[0] <= h - 2 and 1 <= ij[1] <= w - 2)
    bd = random.choice(totuple(reminds))
    go = fill(go, 2, {rd})
    go = fill(go, 1, {bd})
    ngd = unifint(diff_lb, diff_ub, (1, 8))
    gd = random.sample(totuple(neighbors(rd)), ngd)
    nod = unifint(diff_lb, diff_ub, (1, 8))
    od = random.sample(totuple(neighbors(bd)), nod)
    go = fill(go, 3, gd)
    go = fill(go, 7, od)
    gdmapper = {d: (3, position({rd}, {d})) for d in gd}
    odmapper = {d: (7, position({bd}, {d})) for d in od}
    mpr = {**gdmapper, **odmapper}
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(diff_lb, diff_ub, (1, ub))
    gi = tuple(e for e in go)
    fullinds = asindices(gi)
    for _ in range(ndist):
        options = []
        for loc, (col, direc) in mpr.items():
            ii, jj = add(loc, direc)
            if (ii, jj) in fullinds and gi[ii][jj] == bgc:
                options.append((loc, col, direc))
        if len(options) == 0:
            break
        loc, col, direc = random.choice(options)
        del mpr[loc]
        newloc = add(loc, direc)
        mpr[newloc] = (col, direc)
        gi = fill(gi, bgc, {loc})
        gi = fill(gi, col, {newloc})
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading taken from the rotate-and-slide solution: take one satellite colour
    at a time with its anchor (7 -> the 1, then 3 -> the 2). Sweep around that
    anchor one compass direction after another. On each ray, the single dot lying
    on it slides straight back along the ray until it touches the anchor."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The generator always draws the background from outside {1, 2, 3, 7}.
    cnt = Counter(I.flatten().tolist())
    cands = [c for c in cnt if c not in (1, 2, 3, 7)]
    bgc = max(cands, key=lambda c: cnt[c]) if cands else 0

    sgn = lambda v: (v > 0) - (v < 0)
    COMPASS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def anchor(col):
        p = np.argwhere(I == col)
        return (int(p[0][0]), int(p[0][1])) if len(p) else None

    # Group the work by anchor colour, then by compass ray (not raster order).
    dots = []
    for col, anc_col in ((7, 1), (3, 2)):
        ctr = anchor(anc_col)
        if ctr is None:
            continue
        by_dir = {}
        for cell in np.argwhere(I == col):
            r, c = int(cell[0]), int(cell[1])
            u = (sgn(r - ctr[0]), sgn(c - ctr[1]))
            if u == (0, 0):
                continue
            by_dir.setdefault(u, []).append((r, c))
        for u in COMPASS:
            for (r, c) in sorted(by_dir.get(u, []),
                                 key=lambda p: max(abs(p[0] - ctr[0]), abs(p[1] - ctr[1]))):
                dots.append({"src": (r, c), "dst": (ctr[0] + u[0], ctr[1] + u[1]), "col": col})

    frozen = {d["src"] for d in dots if d["src"] == d["dst"]}   # already home
    pending = [d for d in dots if d["src"] != d["dst"]]

    # Keep the colour-then-compass order. Only defer a slide when its landing
    # cell is still held by a dot that has not moved yet.
    order = []
    while pending:
        occupied = frozen | {d["src"] for d in pending}
        pick = None
        for d in pending:
            if d["dst"] not in (occupied - {d["src"]}):
                pick = d
                break
        if pick is None:
            pick = pending[0]
        pending.remove(pick)
        order.append(pick)

    ops, sels = [], []
    for d in order:
        (sr, sc), (tr, tc) = d["src"], d["dst"]
        vr, vc = tr - sr, tc - sc
        steps = []
        for i in range(max(abs(vr), abs(vc))):          # diagonal rays alternate the axes
            if i < abs(vr):
                steps.append(20 if vr < 0 else 21)
            if i < abs(vc):
                steps.append(23 if vc < 0 else 22)
        for k, mop in enumerate(steps):
            ops.append(mop)
            # The first Move grabs this dot. Empty selections after it keep the same
            # dot grabbed, so ARCLE restores every cell it passes over.
            sels.append(sel_of([(sr, sc)]) if k == 0 else sel_of([]))
        if bgc != 0:
            # The grab left the dot's starting cell at 0, so paint it back to the canvas colour.
            ops.append(bgc)
            sels.append(sel_of([(sr, sc)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
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
                                f"for task ae3edfdc"
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
                    f"Failed to build a complete episode for task ae3edfdc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae3edfdc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
