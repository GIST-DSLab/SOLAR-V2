"""
ARC Task: 73251a56 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random

VARIANTS = [{"axis": "main"}, {"axis": "anti"}]


def sample_colors(num_examples=None) -> dict:
    """Episode-level color roles + the mirror-axis plan (discrete structural variant)."""
    cols = list(range(10))
    noisec = random.choice(cols)
    pool = [c for c in cols if c != noisec]
    random.shuffle(pool)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "ccols_pool": pool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, ccols_pool=None, axis=None) -> dict:
    import random as _r
    cols = list(range(10))
    if noisec is None:
        noisec = _r.choice(cols)
    if ccols_pool is None:
        ccols_pool = [c for c in cols if c != noisec]
        _r.shuffle(ccols_pool)
    if axis is None:
        axis = _r.choice(["main", "anti"])

    dub = min(30, int(max_h), int(max_w))
    dlb = min(10, dub)

    while True:
        d = unifint(diff_lb, diff_ub, (dlb, dub))
        h, w = d, d
        nsl = unifint(diff_lb, diff_ub, (2, max(2, min(9, h // 2))))
        nsl = max(2, min(nsl, len(ccols_pool), h - 1))
        slopes = [0] + sorted(_r.sample(list(range(1, h - 1)), nsl - 1))
        ccols = list(ccols_pool[:nsl])

        gi = canvas(-1, (h, w))
        inds = asindices(gi)
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            locs = sfilter(inds, lambda ij: slope * ij[1] <= ij[0])
            gi = fill(gi, col, locs)
        ln = connect((0, 0), (d - 1, d - 1))
        gi = fill(gi, ccols[-2], ln)
        obj = asobject(gi)
        obj = sfilter(obj, lambda cij: cij[1][1] >= cij[1][0])
        gi = paint(gi, dmirror(obj))

        cf1 = lambda g: ccols[-2] in palette(toobject(ln, g))
        cf2 = lambda g: len((ofcolor(g, noisec) & frozenset({ij[::-1] for ij in ofcolor(g, noisec)})) - ln) == 0

        ndist = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
        tr = 0
        succ = 0
        maxtr = 10 * ndist
        go = tuple(e for e in gi)
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = _r.randint(1, max(1, min(5, h - 2)))
            ow = _r.randint(1, max(1, min(5, w - 2)))
            loci = _r.randint(1, h - oh - 1)
            locj = _r.randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi != go:
            break

    if axis == "main":
        rotf = _r.choice((identity, rot180))
    else:
        rotf = _r.choice((rot90, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: the grid is symmetric about one of its diagonals, except for blobs of a single
    'noise' colour.  Route:
      1) actually PERFORM the reflection on the whole grid
         (transpose = Rotate90(CCW) + FlipV ; anti-transpose = Rotate270(CW) + FlipV).
         This carries every clean half onto the half the noise covered.
      2) the reflection drags the noise onto the mirror side, so repaint those mirrored
         cells back to the values the input still shows there (one op per region/colour).
      3) restore the symmetry axis itself where noise sat on it (it is one uniform colour,
         the colour of its clean corner).
    """
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    n, m = I.shape
    ops, sels = [], []
    full = [0, 0, n - 1, m - 1]          # bbox == exactly the whole grid (background included)

    def anti(a):
        return a[::-1, ::-1].T

    # ---- which diagonal is the mirror axis (measured, not assumed) ----
    axis = None
    if n == m:
        cands = []
        if np.array_equal(O, O.T):
            cands.append('main')
        if np.array_equal(O, anti(O)):
            cands.append('anti')
        if not cands:
            cands = ['main', 'anti']
        if len(cands) > 1:
            bad_main = sum(1 for i in range(n) if I[i, i] != I[0, 0])
            bad_anti = sum(1 for i in range(n) if I[i, n - 1 - i] != I[0, n - 1])
            axis = 'main' if bad_main <= bad_anti else 'anti'
        else:
            axis = cands[0]

    cur = I.copy()

    if axis is not None:
        if axis == 'main':
            mir = lambda p: (p[1], p[0])
            mirror_arr = lambda a: a.T.copy()
            rot_op = 24                                  # Rotate90 (CCW) then FlipV -> transpose
            axis_cells = [(i, i) for i in range(n)]
            corner = int(I[0, 0])
        else:
            mir = lambda p: (n - 1 - p[1], n - 1 - p[0])
            mirror_arr = lambda a: anti(a).copy()
            rot_op = 25                                  # Rotate270 (CW) then FlipV -> anti-transpose
            axis_cells = [(i, n - 1 - i) for i in range(n)]
            corner = int(I[0, n - 1])

        # ---- noise colour: the colour the input carries where input and output disagree ----
        cnt = {}
        for r in range(n):
            for c in range(m):
                if I[r, c] != O[r, c]:
                    v = int(I[r, c])
                    cnt[v] = cnt.get(v, 0) + 1
        noisec = max(cnt, key=lambda k: cnt[k]) if cnt else None

        FIX = set()
        if noisec is not None:
            for r in range(n):
                for c in range(m):
                    p = (r, c)
                    q = mir(p)
                    if p == q:
                        continue
                    if I[r, c] == noisec and I[q] != noisec:
                        FIX.add(p)

        # ---- 1) the reflection itself ----
        if FIX:
            ops.append(rot_op); sels.append(full)
            ops.append(27);     sels.append(full)
            cur = mirror_arr(cur)

            # ---- 2) repaint the half the reflection carried noise onto ----
            REP = sorted({mir(p) for p in FIX})
            repset = set(REP)
            seen = set()
            comps = []
            for p in REP:
                if p in seen:
                    continue
                stack = [p]; seen.add(p); comp = []
                while stack:
                    q = stack.pop(); comp.append(q)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        t = (q[0] + dr, q[1] + dc)
                        if t in repset and t not in seen:
                            seen.add(t); stack.append(t)
                comps.append(sorted(comp))
            comps.sort(key=lambda cp: (min(x for x, _ in cp), min(y for _, y in cp)))

            planned = []
            for comp in comps:
                bycol = {}
                for p in comp:
                    v = int(I[p])
                    if int(cur[p]) != v:
                        bycol.setdefault(v, []).append(p)
                for v in sorted(bycol):
                    planned.append((v, bycol[v]))

            if len(planned) > 40:                      # keep the trajectory tractable
                bycol = {}
                for p in REP:
                    v = int(I[p])
                    if int(cur[p]) != v:
                        bycol.setdefault(v, []).append(p)
                planned = [(v, bycol[v]) for v in sorted(bycol)]

            for v, cells in planned:
                if not cells:
                    continue
                ops.append(int(v)); sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = v

        # ---- 3) the axis line is one uniform colour; restore where noise covered it ----
        axfix = [p for p in axis_cells if int(cur[p]) != corner]
        if axfix:
            ops.append(int(corner)); sels.append(sel_of(axfix))
            for p in axfix:
                cur[p] = corner

    # ---- safety net (should be empty) ----
    rem = {}
    for r in range(n):
        for c in range(m):
            if int(cur[r, c]) != int(O[r, c]):
                rem.setdefault(int(O[r, c]), []).append((r, c))
    for v in sorted(rem):
        ops.append(int(v)); sels.append(sel_of(rem[v]))
        for p in rem[v]:
            cur[p] = v

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 73251a56"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 73251a56"
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
                                f"for task 73251a56"
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
                    f"Failed to build a complete episode for task 73251a56 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"73251a56-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
