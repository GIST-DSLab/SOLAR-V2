"""
ARC Task: e73095fd (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # generator draws bgc/fgc from interval(0,10) minus 4 (4 is reserved for the fill color)
    cols = [c for c in range(10) if c != 4]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    while True:
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h + 2)))   # trim() removes 2
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w + 2)))
        gi = canvas(bgc, (h, w))
        nsplits = unifint(diff_lb, diff_ub, (2, max(2, min(h, w) // 3)))
        for k in range(nsplits):
            objs = objects(gi, T, F, F)
            objs = colorfilter(objs, bgc)
            objs = apply(toindices, objs)
            hobjs = sfilter(objs, lambda o: height(o) > 6)
            wobjs = sfilter(objs, lambda o: width(o) > 6)
            if len(hobjs) == 0 and len(wobjs) == 0:
                break
            cgroups = [(g, ax) for g, ax in zip([hobjs, wobjs], [0, 1]) if len(g) > 0]
            g, ax = choice(cgroups)
            obj = choice(totuple(g))
            ulci, ulcj = ulcorner(obj)
            oh, ow = shape(obj)
            if ax == 0:
                iloc = randint(ulci + 3, ulci + oh - 3)
                bar = sfilter(obj, lambda ij: ij[0] == iloc)
            else:
                jloc = randint(ulcj + 3, ulcj + ow - 3)
                bar = sfilter(obj, lambda ij: ij[1] == jloc)
            gi = fill(gi, fgc, bar)
        copts = sfilter(
            ofcolor(gi, fgc),
            lambda ij: len(sfilter(toobject(dneighbors(ij), gi), lambda cij: cij[0] == fgc)) > 2
        )
        copts = sfilter(
            copts,
            lambda ij: len(sfilter(toobject(outbox(outbox({ij})), gi), lambda cij: cij[0] == fgc)) in {3, 4}
        )
        if len(copts) == 0:
            continue
        noccs = unifint(diff_lb, diff_ub, (1, len(copts)))
        noccs = unifint(diff_lb, diff_ub, (noccs, len(copts)))
        occs = sample(totuple(copts), noccs)
        go = tuple(e for e in gi)
        forb = set()
        for occ in occs:
            ulci, ulcj = decrement(occ)
            lrci, lrcj = increment(occ)
            if len(sfilter(toobject(box({(ulci, ulcj), (lrci, lrcj)}), gi), lambda cij: cij[0] == fgc)) in {3, 4}:
                boptions = []
                for ulcioffs in [-2, -1, 0]:
                    for ulcjoffs in [-2, -1, 0]:
                        for lrcioffs in [0, 1, 2]:
                            for lrcjoffs in [0, 1, 2]:
                                bx = box({(ulci + ulcioffs, ulcj + ulcjoffs), (lrci + lrcioffs, lrcj + lrcjoffs)})
                                bxobj = toobject(bx, gi)
                                if len(sfilter(toobject(bxobj, gi), lambda cij: cij[0] == fgc)) in {3, 4} and \
                                   len(sfilter(toobject(outbox(bxobj), gi), lambda cij: cij[0] == fgc)) in {3, 4}:
                                    boptions.append(bx)
                boptions = sfilter(boptions, lambda bx: len(backdrop(bx) & forb) == 0)
                if len(boptions) > 0:
                    bx = choice(boptions)
                    bd = backdrop(bx)
                    gi = fill(gi, bgc, bd)
                    gi = fill(gi, fgc, bx)
                    go = fill(go, 4, bd)
                    go = fill(go, fgc, bx)
                    forb |= bd
        gi = trim(gi)
        go = trim(go)
        if 4 in palette(go):
            break
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I alone): the grid holds background + straight foreground bars, and a
    few small closed rectangular BOXES sitting on bar junctions.  A box is recognised by
    its hollow interior: a background component that
      * is a solid rectangle,
      * is at most 5x5 (a box interior can never be larger by construction),
      * has every in-grid cell of its one-cell surrounding ring in the foreground color
        (that ring IS the box outline), and
      * has no foreground cell straight outside the ring's four corners -- bars that merely
        cross each other keep running past their corner, a real box does not.
    Each such enclosed interior gets flood-filled with color 4.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    rest = [c for c in cnt if c != bgc]
    fgc = rest[0] if rest else bgc

    ops, sels = [], []
    seen = np.zeros((hi, wi), dtype=bool)

    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] != bgc or seen[sr, sc]:
                continue
            # background connected component
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == bgc:
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            r0 = min(p[0] for p in comp); r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp); c1 = max(p[1] for p in comp)
            rh, rw = r1 - r0 + 1, c1 - c0 + 1

            if len(comp) != rh * rw:          # not a solid rectangle -> not a box interior
                continue
            if rh > 5 or rw > 5:              # box interiors are at most 5x5
                continue

            # surrounding ring must be the box outline
            ring_ok = True
            for y in range(r0 - 1, r1 + 2):
                for x in range(c0 - 1, c1 + 2):
                    if y in (r0 - 1, r1 + 1) or x in (c0 - 1, c1 + 1):
                        if 0 <= y < hi and 0 <= x < wi and I[y, x] != fgc:
                            ring_ok = False
                            break
                if not ring_ok:
                    break
            if not ring_ok:
                continue

            # nothing may continue straight outward from the ring corners (that means bars)
            outer = [(r0 - 2, c0 - 1), (r0 - 1, c0 - 2),
                     (r0 - 2, c1 + 1), (r0 - 1, c1 + 2),
                     (r1 + 2, c0 - 1), (r1 + 1, c0 - 2),
                     (r1 + 2, c1 + 1), (r1 + 1, c1 + 2)]
            if any(0 <= y < hi and 0 <= x < wi and I[y, x] == fgc for y, x in outer):
                continue

            # enclosed interior -> flood it with 4
            ops.append(14)
            sels.append([r0, c0, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task e73095fd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e73095fd"
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
                                f"for task e73095fd"
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
                    f"Failed to build a complete episode for task e73095fd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e73095fd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
