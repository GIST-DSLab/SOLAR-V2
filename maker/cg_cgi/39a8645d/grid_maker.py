"""
ARC Task: 39a8645d (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule only depends on shape-frequency and the background colour;
    # the object colours play no role, so only bgc is fixed for the episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = _unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    oh = random.randint(2, 4)
    ow = random.randint(2, 4)
    remcols = [c for c in cols if c != bgc]
    nobjs = _unifint(diff_lb, diff_ub, (1, oh + ow))
    nobjs = min(nobjs, len(remcols) - 1)
    ccols = random.sample(remcols, nobjs + 1)
    mxcol = ccols[0]
    rcols = ccols[1:]
    maxnocc = _unifint(diff_lb, diff_ub, (nobjs + 2, max(nobjs + 2, (h * w) // 16)))
    maxtr = 10 * maxnocc

    def nbr8(cell):
        r, c = cell
        return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def nbr4(cell):
        r, c = cell
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def shape_of(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    def normalize(cells):
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    allobjs = []
    for _ in range(nobjs + 1):
        while True:
            ncells = random.randint(oh + ow - 1, oh * ow)
            cobj = {random.choice(sorted(bounds))}
            while shape_of(cobj) != (oh, ow) and len(cobj) < ncells:
                frontier = set()
                for cc in cobj:
                    frontier |= nbr8(cc)
                cand = sorted((bounds - cobj) & frontier)
                cobj.add(random.choice(cand))
            if frozenset(cobj) not in allobjs:
                break
        allobjs.append(frozenset(cobj))
    mcobj = normalize(allobjs[0])
    remobjs = [normalize(o) for o in allobjs[1:]]
    mxcount = 0
    remcount = {i: 0 for i in range(len(remobjs))}
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    colcells = {}
    tr = 0
    succ = 0
    while tr < maxtr and succ < maxnocc:
        tr += 1
        cand_idx = [i for i, cnt in remcount.items() if cnt + 1 < mxcount]
        if len(cand_idx) == 0 or random.randint(0, 100) / 100 > diff_lb:
            obj, col, idx = mcobj, mxcol, None
        else:
            idx = random.choice(cand_idx)
            obj, col = remobjs[idx], rcols[idx]
        cands = [(r, c) for (r, c) in inds if r <= h - oh and c <= w - ow]
        if not cands:
            break
        lr, lc = random.choice(cands)
        plcd = {(r + lr, c + lc) for r, c in obj}
        forb = set()
        for cc in colcells.get(col, set()):
            forb |= nbr8(cc)
        if plcd <= (inds - forb):
            succ += 1
            rem = set(plcd)
            for cc in plcd:
                rem |= nbr4(cc)
            inds -= rem
            for r, c in plcd:
                gi[r][c] = col
            colcells.setdefault(col, set()).update(plcd)
            if idx is None:
                mxcount += 1
            else:
                remcount[idx] += 1
    ohh = max(r for r, _ in mcobj) + 1
    oww = max(c for _, c in mcobj) + 1
    go = [[bgc] * oww for _ in range(ohh)]
    for r, c in mcobj:
        go[r][c] = mxcol
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: the answer is the shape that is repeated most often in the input.
    # The code-golf entry counts the most frequent fixed-size window rather than the
    # most frequent object. Either way the answer is the bounding box of one copy
    # of that shape, so both readings give the same operation: crop the canvas to
    # it. Nothing changes here.
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # same-colour 8-connected non-background objects
    seen = [[False] * wi for _ in range(hi)]
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r][c] or int(I[r, c]) == bgc:
                continue
            col = int(I[r, c])
            seen[r][c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr][nc] \
                                and int(I[nr, nc]) == col:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            comps.append((col, cells))

    groups = {}
    for col, cells in comps:
        r0 = min(x for x, _ in cells)
        c0 = min(y for _, y in cells)
        key = (col, frozenset((x - r0, y - c0) for x, y in cells))
        groups.setdefault(key, []).append((r0, c0))

    (col, norm), occs = max(groups.items(), key=lambda kv: len(kv[1]))
    oh = max(x for x, _ in norm) + 1
    ow = max(y for _, y in norm) + 1

    # prefer an occurrence whose bounding box holds only that object and background
    r0, c0 = occs[0]
    for (rr, cc) in occs:
        if all(int(I[rr + i, cc + j]) == (col if (i, j) in norm else bgc)
               for i in range(oh) for j in range(ow)):
            r0, c0 = rr, cc
            break

    ops, sels = [], []
    # crop the canvas to the chosen occurrence's bounding box (a full rectangle)
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])

    # fallback only: if no copy had a clean box, paint over any stray cells that
    # are not part of the object (judged from the cropped input) with background
    stray = [(i, j) for i in range(oh) for j in range(ow)
             if (i, j) not in norm and int(I[r0 + i, c0 + j]) != bgc]
    if stray:
        ops.append(bgc); sels.append(sel_of(stray))

    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 39a8645d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 39a8645d"
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
                                f"for task 39a8645d"
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
                    f"Failed to build a complete episode for task 39a8645d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"39a8645d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
