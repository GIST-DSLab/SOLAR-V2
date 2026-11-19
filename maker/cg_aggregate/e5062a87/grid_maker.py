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


def generate(diff_lb, diff_ub, max_h=30, max_w=30, eligcol=None, objc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if eligcol is None or objc is None:
        eligcol, objc = random.sample(list(range(1, 10)), 2)
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    gi = [[eligcol] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]

    sp = random.choice(inds)
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (3, 9))
    for _ in range(ncells - 1):
        cand = set()
        for (r, c) in obj:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in obj:
                        cand.add((rr, cc))
        obj.add(random.choice(sorted(cand)))
    mr = min(r for r, _ in obj)
    mc = min(c for _, c in obj)
    obj = {(r - mr, c - mc) for r, c in obj}
    oh = max(r for r, _ in obj) + 1
    ow = max(c for _, c in obj) + 1

    nnoise = unifint(diff_lb, diff_ub, (int(0.2 * h * w), int(0.5 * h * w)))
    for (r, c) in random.sample(inds, nnoise):
        gi[r][c] = 0

    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    for k in range(noccs):
        li = random.randint(0, h - oh)
        lj = random.randint(0, w - ow)
        col = objc if k == noccs - 1 else 0
        for (r, c) in obj:
            gi[li + r][lj + c] = col

    go = [row[:] for row in gi]
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            if all(gi[i + r][j + c] == 0 for r, c in obj):
                for r, c in obj:
                    go[i + r][j + c] = objc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule: the single shape drawn in the marker colour is a stencil; every place
    in I where that exact shape sits as a hole of colour 0 is filled with the
    marker colour.  Same reading as the golf entry (pattern-match the marked
    shape against zeros, fill the matches), so the units are the same: one
    Color op per hidden copy of the shape.  Copies wholly covered by other
    copies are dropped (they would add no visible change).
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
        oh = max(dr for dr, _ in norm) + 1
        ow = max(dc for _, dc in norm) + 1

        occs = []
        for i in range(hi - oh + 1):
            for j in range(wi - ow + 1):
                if all(I[i + dr, j + dc] == 0 for dr, dc in norm):
                    occs.append((i, j))
        occs.sort(key=lambda p: (abs(p[0] - r0) + abs(p[1] - c0), p[0], p[1]))
        cellsets = {p: frozenset((p[0] + dr, p[1] + dc) for dr, dc in norm) for p in occs}

        # drop copies whose cells are entirely covered by the other copies
        kept = list(occs)
        changed = True
        while changed:
            changed = False
            for p in reversed(kept):
                others = set()
                for q in kept:
                    if q != p:
                        others |= cellsets[q]
                if cellsets[p] <= others:
                    kept.remove(p)
                    changed = True
                    break

        for p in kept:
            ops.append(int(objc))
            sels.append(sel_of(sorted(cellsets[p])))

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
