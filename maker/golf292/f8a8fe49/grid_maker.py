"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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

import numpy as np

from maker.sel_helpers import sel_of  # noqa: F401  (bbox selections below are full rectangles by intent)


VARIANTS = [
    {"vertical": False},   # box lines are rows
    {"vertical": True},    # box lines are columns (dmirror of the generator's layout)
]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, boxc=None,
             vertical=None) -> dict:
    from dsl import (canvas, connect, backdrop, fill, shift, hmirror, insert,
                     sfilter, lowermost, totuple, asobject, paint, dmirror)
    from utils import unifint

    if bgc is None or objc is None or boxc is None:
        bgc, objc, boxc = random.sample(range(10), 3)
    if vertical is None:
        vertical = random.choice([False, True])

    # in vertical mode the grid gets transposed at the end, so swap the limits
    lim_h, lim_w = (max_w, max_h) if vertical else (max_h, max_w)
    lim_h = max(10, lim_h)
    lim_w = max(4, lim_w)

    h = unifint(diff_lb, diff_ub, (10, lim_h))
    w = unifint(diff_lb, diff_ub, (4, lim_w))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = random.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(random.sample(totuple(cands), ncells))
        cells = insert(random.choice(totuple(sfilter(cands, lambda ij: ij[0] == lowermost(cands)))), cells)
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if vertical:
        gi = dmirror(gi)
        go = dmirror(go)
    go, gi = gi, go
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule: two parallel box lines enclose a strip. The pattern lying just inside
    each box line is reflected THROUGH that line to the outside, and the inside
    ends up empty. Reflecting about a line is the same as flipping, in place, a
    band centred on that line, so there is one flip per box line.
    Each band is a full rectangle (background included) that can be named from I:
    it covers the interior columns (or rows) and is centred on the box line,
    reaching as deep as the pattern on that side goes.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The generator always leaves the grid's top-left corner empty (it lies
    # outside the box and outside every pattern), so I[0, 0] is the background.
    bgc = int(I[0, 0])

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    def area(col):
        b = bbox(col)
        return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)

    others = sorted({int(v) for v in I.flatten().tolist()} - {bgc})
    boxc = max(others, key=area)            # the box bbox strictly contains the patterns
    objc = [c for c in others if c != boxc][0]

    r0, r1, c0, c1 = bbox(boxc)
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))
    rs, cs = np.where(I == objc)

    ops, sels = [], []
    if horizontal:
        mid = (r0 + r1) / 2.0
        m = max(int(r) for r in rs if r < mid) - r0
        # full rectangle centred on the top box row, interior columns only
        ops.append(27); sels.append([r0 - m, c0 + 1, 2 * m, c1 - c0 - 2])
        m = r1 - min(int(r) for r in rs if r > mid)
        ops.append(27); sels.append([r1 - m, c0 + 1, 2 * m, c1 - c0 - 2])
    else:
        mid = (c0 + c1) / 2.0
        m = max(int(c) for c in cs if c < mid) - c0
        # full rectangle centred on the left box column, interior rows only
        ops.append(26); sels.append([r0 + 1, c0 - m, r1 - r0 - 2, 2 * m])
        m = c1 - min(int(c) for c in cs if c > mid)
        ops.append(26); sels.append([r0 + 1, c1 - m, r1 - r0 - 2, 2 * m])

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
