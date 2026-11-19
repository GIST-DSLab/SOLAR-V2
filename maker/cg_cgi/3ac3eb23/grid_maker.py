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


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))
    if bgc is None:
        bgc = random.choice(range(10))
    mf = random.choice((0, 1, 2, 3))
    hi_cap = max_w if mf in (1, 3) else max_h
    wi_cap = max_h if mf in (1, 3) else max_w
    h = unifint(diff_lb, diff_ub, (3, max(3, hi_cap)))
    w = unifint(diff_lb, diff_ub, (3, max(3, wi_cap)))
    remcols = [c for c in range(10) if c != bgc]
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = list(range(1, w - 1))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if not (locj - 2 <= x < locj + 3)]
        col = random.choice(remcols)
        gi[0, locj] = col
        for p in range(0, h, 2):
            go[p, locj] = col
        for p in range(1, h, 2):
            go[p, locj - 1] = col
            go[p, locj + 1] = col
    # mf: 0 identity, 1 rot90 (CW), 2 rot180, 3 rot270
    k = {0: 0, 1: 3, 2: 2, 3: 1}[mf]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading: the border the seeds sit on, together with the lane next to it,
    forms ONE 2-deep strip unit (seed lane + wing lane, all seeds at once).
    The output is that strip stamped over and over into the grid.
      1. each seed sprouts its two wings in the adjacent lane;
      2. the whole 2-deep border strip is copied as one unit;
      3. it is pasted every 2 lanes inward; an odd leftover lane gets the
         seed lane alone.
    Colour 0 is invisible to Copy/Paste, so a 0-coloured seed's fan is painted
    directly afterwards.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    if examples:
        cnt = Counter()
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    if all(r == 0 for r, _ in seeds):
        u, v, D, L = (1, 0), (0, 1), hi, wi
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D, L = (-1, 0), (0, 1), hi, wi
    elif all(c == 0 for _, c in seeds):
        u, v, D, L = (0, 1), (1, 0), wi, hi
    else:
        u, v, D, L = (0, -1), (1, 0), wi, hi

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    # border lane origin (t measured along v from the grid edge)
    if u == (1, 0):
        base = (0, 0)
    elif u == (-1, 0):
        base = (hi - 1, 0)
    elif u == (0, 1):
        base = (0, 0)
    else:
        base = (0, wi - 1)

    def lane(d):
        return [(base[0] + u[0] * d + v[0] * t, base[1] + u[1] * d + v[1] * t)
                for t in range(L)]

    ops, sels = [], []
    # 1. wings
    for s in seeds:
        col = int(I[s])
        ops.append(col)
        sels.append(sel_of([cell(s, 1, -1), cell(s, 1, 1)]))

    def topleft(cells):
        return (min(p[0] for p in cells), min(p[1] for p in cells))

    use_copy = any(int(I[s]) != 0 for s in seeds) and D > 2
    if use_copy:
        P = D // 2
        unit = lane(0) + lane(1)
        if P > 1 or u[0] + u[1] > 0:
            # whole 2-deep border strip (a full rectangle) is the unit
            ops.append(29)
            sels.append(sel_of(unit))
        for k in range(1, P):
            tl = topleft(lane(2 * k) + lane(2 * k + 1))
            ops.append(30)
            sels.append(sel_of([tl]))
        if D % 2 == 1:
            last = lane(D - 1)
            if u[0] + u[1] > 0:
                # inward direction is +: stamp the 2-deep unit; its wing lane
                # falls past the far edge and is clipped
                ops.append(30)
                sels.append(sel_of([topleft(last)]))
            else:
                # inward is -: the leftover lane is the seed lane alone
                ops.append(29)
                sels.append(sel_of(lane(0)))
                ops.append(30)
                sels.append(sel_of([topleft(last)]))
    # colour-0 seeds (invisible to Copy/Paste) or tiny strips: draw fan directly
    for s in seeds:
        col = int(I[s])
        if use_copy and col != 0:
            continue
        cells = []
        for d in range(2, D):
            for t in ((0,) if d % 2 == 0 else (-1, 1)):
                cells.append(cell(s, d, t))
        if cells:
            ops.append(col)
            sels.append(sel_of(cells))

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
