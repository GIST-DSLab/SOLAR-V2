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


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    rem = [c for c in range(10) if c != bgc]
    random.shuffle(rem)
    return {"bgc": bgc, "ccols": rem}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _nbrs8(r, c):
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def _nbrs4(r, c):
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _norm(cells):
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shape(cells):
    return (max(r for r, _ in cells) - min(r for r, _ in cells) + 1,
            max(c for _, c in cells) - min(c for _, c in cells) + 1)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols=None, **kw) -> dict:
    if ccols is None:
        ccols = [c for c in range(10) if c != bgc]
        random.shuffle(ccols)
    h = _unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = _unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    oh = random.randint(2, 4)
    ow = random.randint(2, 4)
    nobjs = _unifint(diff_lb, diff_ub, (1, min(oh + ow, len(ccols) - 1)))
    cc = list(ccols[:nobjs + 1])
    mxcol, rcols = cc[0], cc[1:]
    maxnocc = _unifint(diff_lb, diff_ub, (nobjs + 2, max(nobjs + 2, (h * w) // 16)))
    tr = 0
    maxtr = 10 * maxnocc
    succ = 0
    allobjs = []
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    for k in range(nobjs + 1):
        while True:
            ncells = random.randint(oh + ow - 1, oh * ow)
            cobj = {random.choice(sorted(bounds))}
            while _shape(cobj) != (oh, ow) and len(cobj) < ncells:
                fr = set()
                for (r, c) in cobj:
                    fr |= _nbrs8(r, c)
                cobj.add(random.choice(sorted((bounds - cobj) & fr)))
            if frozenset(cobj) not in allobjs:
                break
        allobjs.append(frozenset(cobj))
    mcobj = _norm(allobjs[0])
    remobjs = [_norm(o) for o in allobjs[1:]]
    mxobjcounter = 0
    remobjcounter = {robj: 0 for robj in remobjs}
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    forbid = {c: set() for c in cc}
    while tr < maxtr and succ < maxnocc:
        tr += 1
        candobjs = [robj for robj, cnt in remobjcounter.items() if cnt + 1 < mxobjcounter]
        if len(candobjs) == 0 or random.randint(0, 100) / 100 > diff_lb:
            obj, col = mcobj, mxcol
        else:
            obj = random.choice(candobjs)
            col = rcols[remobjs.index(obj)]
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            break
        li, lj = random.choice(cands)
        plcd = {(r + li, c + lj) for r, c in obj}
        if plcd <= (inds - forbid[col]):
            succ += 1
            rm = set(plcd)
            for (r, c) in plcd:
                rm |= _nbrs4(r, c)
                forbid[col] |= _nbrs8(r, c)
            inds -= rm
            for (r, c) in plcd:
                gi[r][c] = col
            if obj in remobjcounter:
                remobjcounter[obj] += 1
            else:
                mxobjcounter += 1
    if mxobjcounter == 0:
        raise ValueError("no answer object placed")
    if any(v >= mxobjcounter for v in remobjcounter.values()):
        raise ValueError("answer shape not the unique mode")
    sh = _shape(mcobj)
    go = [[bgc] * sh[1] for _ in range(sh[0])]
    for r, c in mcobj:
        go[r][c] = mxcol
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background is fixed for the episode: the dominant colour over I and every
    # demonstration input (all drawn on the same canvas colour).
    cnt = Counter(I.flatten().tolist())
    for ex in (examples or []):
        cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # Objects: same-colour 8-connected non-background components of I.
    seen = np.zeros((hi, wi), dtype=bool)
    groups = {}
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _nbrs8(cr, cc):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0 = min(x for x, _ in cells); c0 = min(y for _, y in cells)
            key = (col, _norm(cells))
            groups.setdefault(key, []).append((r0, c0))

    # The answer is the most frequently repeated shape (with its colour).
    (col, norm), occs = max(groups.items(), key=lambda kv: len(kv[1]))
    oh = max(x for x, _ in norm) + 1
    ow = max(y for _, y in norm) + 1

    def intruders(rr, cc):
        return [(rr + i, cc + j) for i in range(oh) for j in range(ow)
                if (i, j) not in norm and I[rr + i, cc + j] != bgc]

    # Prefer a copy whose bounding box contains only the object and background.
    r0, c0 = min(occs, key=lambda p: (len(intruders(*p)), p))
    junk = intruders(r0, c0)

    ops, sels = [], []
    # Crop the canvas to that copy's bounding box (full rectangle: object + background).
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])
    # Any neighbouring object poking into the box is not part of the shape: clear it.
    if junk:
        ops.append(bgc); sels.append(sel_of([(r - r0, c - c0) for r, c in junk]))
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
