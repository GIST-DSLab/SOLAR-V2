"""
ARC Task: e5062a87 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    eligcol, objc = random.sample(cols, 2)
    return {"eligcol": eligcol, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, eligcol=None, objc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        lo_i = int(round(lo))
        hi_i = int(round(hi))
        if hi_i < lo_i:
            hi_i = lo_i
        return random.randint(lo_i, hi_i)

    cols = list(range(1, 10))
    if eligcol is None or objc is None or eligcol == objc:
        eligcol, objc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[eligcol] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    indset = set(inds)

    def neighbors(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    sp = random.choice(inds)
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (3, 9))
    for _ in range(ncells - 1):
        nb = set()
        for p in obj:
            nb |= neighbors(p)
        cands = sorted((indset - obj) & nb)
        obj.add(random.choice(cands))
    r0 = min(i for i, _ in obj)
    c0 = min(j for _, j in obj)
    obj = {(i - r0, j - c0) for i, j in obj}
    nnoise = unifint(diff_lb, diff_ub, (int(0.2 * h * w), int(0.5 * h * w)))
    for (i, j) in random.sample(inds, nnoise):
        gi[i][j] = 0
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    for k in range(noccs):
        li = random.randint(0, h - oh)
        lj = random.randint(0, w - ow)
        colr = objc if k == noccs - 1 else 0
        for (a, b) in obj:
            gi[li + a][lj + b] = colr
    go = [row[:] for row in gi]
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            if all(gi[i + a][j + b] == 0 for a, b in obj):
                for a, b in obj:
                    go[i + a][j + b] = objc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule (from I + demonstrations only):
      * the single shape drawn in the marker colour is a stencil;
      * slide the stencil over every offset of the grid (as the golf entry does);
        wherever every stencil cell lands on a 0-hole, that hole is the hidden
        copy and gets painted in the marker colour.
    Both readings agree on the unit (one stencil placement = one paint op).
    Overlapping placements whose cells are entirely covered by other placements
    add nothing of their own, so they are dropped (computed from I alone).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    cnt = Counter(I.flatten().tolist())
    objc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if examples:
        seen = set()
        for pair in examples:
            try:
                ei, eo = np.asarray(pair[0], dtype=int), np.asarray(pair[1], dtype=int)
            except Exception:
                continue
            if ei.shape != eo.shape:
                continue
            added = set(eo[ei != eo].tolist())
            if len(added) == 1:
                seen.add(added.pop())
        if len(seen) == 1 and seen.issubset(set(cnt.keys())):
            objc = next(iter(seen))

    ops, sels = [], []
    marker = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == objc]
    if marker:
        r0 = min(r for r, _ in marker)
        c0 = min(c for _, c in marker)
        norm = sorted((r - r0, c - c0) for r, c in marker)
        oh = max(a for a, _ in norm) + 1
        ow = max(b for _, b in norm) + 1

        # slide the stencil over the grid in scan order of offsets
        occs = []
        for i in range(hi - oh + 1):
            for j in range(wi - ow + 1):
                if all(I[i + a, j + b] == 0 for a, b in norm):
                    occs.append(frozenset((i + a, j + b) for a, b in norm))

        # drop placements fully covered by the remaining others (each kept
        # placement then paints at least one cell no other placement paints)
        keep = list(occs)
        changed = True
        while changed:
            changed = False
            cover = Counter()
            for s in keep:
                cover.update(s)
            for idx in range(len(keep) - 1, -1, -1):
                s = keep[idx]
                if all(cover[c] > 1 for c in s):
                    keep.pop(idx)
                    changed = True
                    break

        for s in keep:
            ops.append(int(objc))
            sels.append(sel_of(sorted(s)))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e5062a87"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e5062a87"
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
                                f"for task e5062a87"
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
                    f"Failed to build a complete episode for task e5062a87 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e5062a87-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
