"""
ARC Task: 0520fde7 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(1, 10) if c != 2]
    barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    v = random.uniform(lo, hi)
    return max(a, min(b, int(round(v))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, barcol=8, cola=1, colb=3,
             transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    if transpose:
        lim_h, lim_w = max_w, max_h
    else:
        lim_h, lim_w = max_h, max_w
    h = _unifint(diff_lb, diff_ub, (2, max(2, min(30, lim_h))))
    w = _unifint(diff_lb, diff_ub, (2, max(2, min(14, (lim_w - 1) // 2))))
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2
    deva = _unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    devb = _unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    a = random.sample(inds, numa)
    b = random.sample(inds, numb)
    gia = np.zeros((h, w), dtype=int)
    gib = np.zeros((h, w), dtype=int)
    for (i, j) in a:
        gia[i, j] = cola
    for (i, j) in b:
        gib[i, j] = colb
    gbar = np.full((h, 1), barcol, dtype=int)
    gi = np.concatenate([gia, gbar, gib], axis=1)
    go = np.zeros((h, w), dtype=int)
    for (i, j) in set(a) & set(b):
        go[i, j] = 2
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    """Same rule, different carving: crop to the first half FIRST (the second
    half stays readable from the input), turn the whole first-half colour into
    2 in one colour-level stroke, then walk the half line by line clearing the
    marks whose mirrored partner in the input's second half is empty."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    whalf, hhalf = wi // 2, hi // 2
    vertical = len(set(I[:, whalf].tolist())) == 1
    if vertical:
        ch, cw = hi, whalf
        A = I[:, :whalf]
        B = I[:, wi - whalf:]
    else:
        ch, cw = hhalf, wi
        A = I[:hhalf, :]
        B = I[hi - hhalf:, :]
    # 1. crop the canvas to the first half (bbox = exactly that full rectangle)
    ops.append(33); sels.append([0, 0, ch - 1, cw - 1])
    # 2. all cells of the first half's colour become 2 (one colour at once)
    marked = [(r, c) for r in range(ch) for c in range(cw) if A[r, c] != 0]
    if marked:
        ops.append(2); sels.append(sel_of(marked))
    # 3. line by line: clear marks whose partner in the second half is empty
    for r in range(ch):
        row = [(r, c) for c in range(cw) if A[r, c] != 0 and B[r, c] == 0]
        if row:
            ops.append(0); sels.append(sel_of(row))
    ops.append(34); sels.append([0, 0, ch - 1, cw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0520fde7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0520fde7"
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
                                f"for task 0520fde7"
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
                    f"Failed to build a complete episode for task 0520fde7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0520fde7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
