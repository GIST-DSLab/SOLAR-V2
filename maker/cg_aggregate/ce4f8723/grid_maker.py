"""
ARC Task: ce4f8723 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    cola = random.choice(rem)
    colb = random.choice([c for c in rem if c != cola])
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barcol": barcol, "cola": cola, "colb": colb,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, cola, colb,
             transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))

    def unif(lo, hi):
        hi = max(lo, hi)
        a = lo + diff_lb * (hi - lo)
        b = lo + diff_ub * (hi - lo)
        return random.randint(int(round(a)), int(round(b)))

    if transpose:
        h = unif(2, min(30, max_w))
        w = unif(2, min(14, (max_h - 1) // 2))
    else:
        h = unif(2, min(30, max_h))
        w = unif(2, min(14, (max_w - 1) // 2))
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2
    deva = unif(0, mp) * random.choice((1, -1))
    devb = unif(0, mp) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    a = random.sample(inds, numa)
    b = random.sample(inds, numb)
    gia = [[bgc] * w for _ in range(h)]
    gib = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for i, j in a:
        gia[i][j] = cola
        go[i][j] = 3
    for i, j in b:
        gib[i][j] = colb
        go[i][j] = 3
    gi = [gia[i] + [barcol] + gib[i] for i in range(h)]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Separator bar: the uniform middle row (horizontal split) or middle column.
    horiz = (hi % 2 == 1) and len(set(I[hi // 2].tolist())) == 1
    if horiz:
        k = hi // 2
        h1, w1 = k, wi
        off = (k + 1, 0)          # origin of the second half
    else:
        k = wi // 2
        h1, w1 = hi, k
        off = (0, k + 1)
    half1 = I[:h1, :w1]
    half2 = I[off[0]:off[0] + h1, off[1]:off[1] + w1]

    # Shared colour = output background; each half's other colour = its marks.
    p1, p2 = set(half1.flatten().tolist()), set(half2.flatten().tolist())
    common = p1 & p2
    s = next(iter(common)) if len(common) == 1 else \
        Counter(I.flatten().tolist()).most_common(1)[0][0]
    m1 = next((c for c in p1 if c != s), s)
    m2 = next((c for c in p2 if c != s), s)

    g = I.copy()

    def paint(color, cells):
        cells = [(r, c) for r, c in cells if g[r, c] != color]
        if cells:
            for r, c in cells:
                g[r, c] = color
            ops.append(color)
            sels.append(sel_of(cells))

    # Reading: the two halves are two transparent layers stacked on each other;
    # a cell ends up marked (3) if either layer marks it.
    # 1) layer 1: every cell of its mark colour becomes 3
    paint(3, [(r, c) for r in range(h1) for c in range(w1) if I[r, c] == m1])
    # 2) layer 2: every cell of its mark colour becomes 3
    paint(3, [(off[0] + r, off[1] + c) for r in range(h1) for c in range(w1)
              if I[off[0] + r, off[1] + c] == m2])
    # 3) make layer 2's background transparent (0) so only its marks transfer
    paint(0, [(off[0] + r, off[1] + c) for r in range(h1) for c in range(w1)
              if I[off[0] + r, off[1] + c] == s])
    # 4) lift layer 2 (whole half rectangle, current grid) and lay it onto layer 1
    if (g[off[0]:off[0] + h1, off[1]:off[1] + w1] != 0).any():
        ops.append(29); sels.append([off[0], off[1], h1 - 1, w1 - 1])  # full half rect
        ops.append(30); sels.append([0, 0, 0, 0])
    # 5) keep only the stacked first half (full rectangle)
    ops.append(33); sels.append([0, 0, h1 - 1, w1 - 1])
    ops.append(34); sels.append([0, 0, h1 - 1, w1 - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ce4f8723"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce4f8723"
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
                                f"for task ce4f8723"
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
                    f"Failed to build a complete episode for task ce4f8723 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce4f8723-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
