"""
ARC Task: 3428a4f5 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def _canvas(v, h, w):
    return [[v] * w for _ in range(h)]


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = 0
    remcols = [c for c in cols if c != bgc]
    barcol = random.choice(remcols)
    remcols = [c for c in remcols if c != barcol]
    cola = random.choice(remcols)
    colb = random.choice(remcols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barcol": barcol, "cola": cola, "colb": colb,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, barcol=8, cola=1, colb=2,
             transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))

    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if transpose:
        # transposed layout: grid is (2w+1) x h
        h = unifint(diff_lb, diff_ub, (1, min(30, max_w)))
        w = unifint(diff_lb, diff_ub, (1, max(1, min(14, (max_h - 1) // 2))))
    else:
        h = unifint(diff_lb, diff_ub, (1, min(30, max_h)))
        w = unifint(diff_lb, diff_ub, (1, max(1, min(14, (max_w - 1) // 2))))
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2
    deva = unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    devb = unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    numa = min(numa, len(inds))
    numb = min(numb, len(inds))
    a = set(random.sample(inds, numa))
    b = set(random.sample(inds, numb))
    gia = _canvas(bgc, h, w)
    gib = _canvas(bgc, h, w)
    for (i, j) in a:
        gia[i][j] = cola
    for (i, j) in b:
        gib[i][j] = colb
    gi = [gia[i] + [barcol] + gib[i] for i in range(h)]
    go = _canvas(bgc, h, w)
    for (i, j) in (a | b) - (a & b):
        go[i][j] = 3
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): lay the two halves over each other
    # cell by cell, and a cell comes out 3 exactly where the halves differ in
    # occupancy. The maker and the golf entry agree on this. The route below
    # builds it in layers: crop to half A, paint A's footprint 3, then lay B's
    # footprint on top. Where B lands on a 3 the two cancel to 0, and where B
    # lands on background it adds a 3.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Find the separator bar from I alone: the middle column (or row) made of
    # one uniform non-zero colour.
    def uniform(v):
        return v.size > 0 and v[0] != 0 and np.all(v == v[0])

    if wi % 2 == 1 and uniform(I[:, wi // 2]):
        ho, wo = hi, wi // 2
        A = I[:, :wo]
        B = I[:, wi - wo:]
    else:
        ho, wo = hi // 2, wi
        A = I[:ho, :]
        B = I[hi - ho:, :]

    # 1. Crop the canvas to half A (bgc == 0, so the transparent copy keeps A as it is).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    a_cells = [(r, c) for r in range(ho) for c in range(wo) if A[r, c] != 0]
    b_cells = [(r, c) for r in range(ho) for c in range(wo) if B[r, c] != 0]
    aset = set(a_cells)

    # 2. A's footprint becomes the marker colour 3.
    if a_cells:
        ops.append(3); sels.append(sel_of(a_cells))

    # 3. Lay B's footprint over it: where B lands on a marked cell, the pair cancels.
    b_on_a = [p for p in b_cells if p in aset]
    if b_on_a:
        ops.append(0); sels.append(sel_of(b_on_a))
    # 4. Where B lands on empty background, it adds a mark.
    b_alone = [p for p in b_cells if p not in aset]
    if b_alone:
        ops.append(3); sels.append(sel_of(b_alone))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3428a4f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3428a4f5"
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
                                f"for task 3428a4f5"
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
                    f"Failed to build a complete episode for task 3428a4f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3428a4f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
