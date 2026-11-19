"""
ARC Task: 88a10436 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = unifint(diff_lb, diff_ub, (0, 2))
    objw = unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    bb = {(i, j) for i in range(objh) for j in range(objw)}
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb.discard(sp)

    def dn(c):
        return {(c[0] - 1, c[1]), (c[0] + 1, c[1]), (c[0], c[1] - 1), (c[0], c[1] + 1)}

    def nb(c):
        return {(c[0] + a, c[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def grow():
        cand = set()
        for c in obj:
            cand |= dn(c)
        return sorted((bb - obj) & cand)

    ncells = unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for _ in range(ncells - 1):
        obj.add(random.choice(grow()))
    while (max(r for r, _ in obj) - min(r for r, _ in obj) + 1 != objh or
           max(c for _, c in obj) - min(c for _, c in obj) + 1 != objw):
        obj.add(random.choice(grow()))
    remcols = [c for c in range(10) if c not in (bgc, fgc)]
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    mr = min(r for r, _ in obj)
    mc = min(c for _, c in obj)
    objc = [(random.choice(ccols), (r - mr, c - mc)) for r, c in obj]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    li = random.randint(0, h - objh)
    lj = random.randint(0, w - objw)
    plc = {(r + li, c + lj) for _, (r, c) in objc}
    for v, (r, c) in objc:
        gi[r + li, c + lj] = v
        go[r + li, c + lj] = v
    allc = {(i, j) for i in range(h) for j in range(w)}
    excl = set(plc)
    for c in plc:
        excl |= nb(c)
    inds = allc - excl
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * ncells))))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if not inds:
            break
        loc = random.choice(sorted(inds))
        p = {(r + loc[0], c + loc[1]) for _, (r, c) in objc}
        if p.issubset(inds):
            for v, (r, c) in objc:
                go[r + loc[0], c + loc[1]] = v
            rs = [r for r, _ in p]
            cs = [c for _, c in p]
            gi[(min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2] = fgc
            succ += 1
            rm = set(p)
            for c in p:
                rm |= dn(c)
            inds = inds - rm
        tr += 1
    if succ == 0:
        raise ValueError("no markers placed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background = canvas colour the generator fills first (dominant colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 4-connected non-background components
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        na, nb_ = a + da, b + db
                        if 0 <= na < hi and 0 <= nb_ < wi and not seen[na, nb_] \
                                and I[na, nb_] != bgc:
                            seen[na, nb_] = True
                            st.append((na, nb_))
                comps.append(sorted(cells))

    if comps:
        # the stamp = the single multi-cell object; markers = isolated single cells
        stamp = max(comps, key=len)
        r0 = min(r for r, _ in stamp)
        c0 = min(c for _, c in stamp)
        oh = max(r for r, _ in stamp) - r0 + 1
        ow = max(c for _, c in stamp) - c0 + 1
        ar, ac = oh // 2, ow // 2               # stamp centre (odd dims)
        zero_rel = [(r - r0, c - c0) for r, c in stamp if I[r, c] == 0]
        has_nz = any(I[r, c] != 0 for r, c in stamp)
        markers = sorted(cl[0] for cl in comps if len(cl) == 1)

        if markers:
            # copy the stamp ONCE as a unit (its own cells only, not its bbox)
            if has_nz:
                ops.append(28)
                sels.append(sel_of(stamp))
            for (mr, mc) in markers:
                tr, tc = mr - ar, mc - ac      # top-left so the stamp centre lands on the marker
                if has_nz:
                    ops.append(30)
                    sels.append([tr, tc, 0, 0])
                if zero_rel:
                    # colour-0 stamp cells are transparent to Copy/Paste:
                    # paint them where the copy lands
                    ops.append(0)
                    sels.append(sel_of([(tr + a, tc + b) for a, b in zero_rel]))

    # submit the whole canvas
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
                        f"num_examples+1 ({num_examples + 1}) for task 88a10436"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 88a10436"
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
                                f"for task 88a10436"
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
                    f"Failed to build a complete episode for task 88a10436 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"88a10436-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
