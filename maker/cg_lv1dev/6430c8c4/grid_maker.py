"""
ARC Task: 6430c8c4 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "acol": acol, "bcol": bcol,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, acol, bcol,
             transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    if transpose:
        # after dmirror: height = 2w+1, width = h
        h_cap = min(30, max_w)
        w_cap = min(14, (max_h - 1) // 2)
    else:
        h_cap = min(30, max_h)
        w_cap = min(14, (max_w - 1) // 2)
    h_cap = max(2, h_cap)
    w_cap = max(2, w_cap)
    h = _unifint(diff_lb, diff_ub, (2, h_cap))
    w = _unifint(diff_lb, diff_ub, (2, w_cap))
    n = h * w
    inds = [(r, c) for r in range(h) for c in range(w)]
    numadev = _unifint(diff_lb, diff_ub, (0, n // 2))
    numbdev = _unifint(diff_lb, diff_ub, (0, n // 2))
    numa = random.choice((numadev, n - numadev))
    numb = random.choice((numadev, n - numbdev))
    numa = min(max(1, numa), n - 1)
    numb = min(max(1, numb), n - 1)
    aset = random.sample(inds, numa)
    bset = random.sample(inds, numb)
    A = np.full((h, w), bgc, dtype=int)
    B = np.full((h, w), bgc, dtype=int)
    for r, c in aset:
        A[r, c] = acol
    for r, c in bset:
        B[r, c] = bcol
    bar = np.full((h, 1), linc, dtype=int)
    gi = np.hstack([A, bar, B])
    go = np.full((h, w), bgc, dtype=int)
    res = set(inds) - set(aset) - set(bset)
    for r, c in res:
        go[r, c] = 3
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _split(I):
    hi, wi = I.shape
    if wi % 2 == 1:
        col = I[:, wi // 2]
        if np.all(col == col[0]):
            k = wi // 2
            return I[:, :k], I[:, k + 1:], "col"
    k = hi // 2
    return I[:k, :], I[k + 1:, :], "row"


def _mark_colour(examples):
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            a, b, _ = _split(ei)
            shared = set(a.flatten().tolist()) & set(b.flatten().tolist())
            if len(shared) == 1:
                bg = shared.pop()
                other = [v for v in set(eo.flatten().tolist()) if v != bg]
                if other:
                    return int(other[0])
    return 3


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    first, second, _ = _split(I)
    shared = set(first.flatten().tolist()) & set(second.flatten().tolist())
    if len(shared) == 1:
        bgc = shared.pop()
    else:
        from collections import Counter
        bgc = Counter(np.concatenate([first.flatten(), second.flatten()]).tolist()).most_common(1)[0][0]
    mark = _mark_colour(examples)
    kh, kw = first.shape

    # Reading: a cell gets the mark only if NEITHER half marks it.
    # Layer it: lay the mark over the whole kept half, then knock out
    # every cell that either half marks (union of the two mark sets).
    ops, sels = [], []
    grid = I.copy()

    # 1. whole kept half -> mark colour (full rectangle, bbox exact)
    ops.append(int(mark)); sels.append([0, 0, kh - 1, kw - 1])
    grid[:kh, :kw] = mark

    # 2. first half's own marks -> background
    a_marks = [(r, c) for r in range(kh) for c in range(kw) if first[r, c] != bgc]
    if a_marks:
        ops.append(int(bgc)); sels.append(sel_of(a_marks))
        for r, c in a_marks:
            grid[r, c] = bgc

    # 3. second half's marks, laid over the same positions -> background
    b_marks = [(r, c) for r in range(kh) for c in range(kw)
               if second[r, c] != bgc and grid[r, c] != bgc]
    if b_marks:
        ops.append(int(bgc)); sels.append(sel_of(b_marks))
        for r, c in b_marks:
            grid[r, c] = bgc

    # 4. keep only the first half (full rectangle -> bbox exact)
    ops.append(33); sels.append([0, 0, kh - 1, kw - 1])
    ops.append(34); sels.append([0, 0, kh - 1, kw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6430c8c4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6430c8c4"
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
                                f"for task 6430c8c4"
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
                    f"Failed to build a complete episode for task 6430c8c4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6430c8c4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
