"""
ARC Task: 496994bd (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback if helper module is unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ---------------------------------------------------------------- 1. colors --

# Discrete structural variants of this task:
#   flag        -> blank half is glued directly (width 2w) or with a 1-col gap (2w+1)
#   do_vmirror  -> content block sits on the LEFT (False) or on the RIGHT (True)
#   do_hmirror  -> whole grid flipped up/down (rows reordered)
VARIANTS = [
    {"flag": f, "do_vmirror": v, "do_hmirror": m}
    for f in (True, False)
    for v in (True, False)
    for m in (True, False)
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))            # generator never uses color 0
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test combo was shown
    return {"bgc": bgc, "instance_plan": plan}


# -------------------------------------------------------------- 2. generate --

def generate(diff_lb, diff_ub, max_h, max_w, bgc,
             flag=None, do_vmirror=None, do_hmirror=None) -> dict:
    cols = interval(1, 10, 1)
    if flag is None:
        flag = choice((True, False))
    if do_vmirror is None:
        do_vmirror = choice((True, False))
    if do_hmirror is None:
        do_hmirror = choice((True, False))

    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    wub = max(3, min(14, (max_w - 1) // 2))      # width becomes 2w or 2w+1
    w = unifint(diff_lb, diff_ub, (3, wub))

    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] < h // 2)))),
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] > h // 2))))
    }
    for kk in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        cands = totuple(bx & dns)
        if len(cands) == 0:
            break
        ch = choice(cands)
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)

    gi = hconcat(gix, canv if flag else hconcat(canvas(bgc, (h, 1)), canv))
    go = hconcat(gix, vmirror(gix) if flag else hconcat(canvas(bgc, (h, 1)), vmirror(gix)))
    if do_vmirror:
        gi = vmirror(gi)
        go = vmirror(go)
    if do_hmirror:
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------- 3. derive_operations --

def derive_operations(I, O):
    """Rule (all parameters measured from I):

    The grid is made of two equal half-blocks of width w = wi // 2 (with an
    optional single background separator column when wi is odd).  One block
    carries all the content, the other is blank background.  The content block
    is duplicated into the blank block and mirrored left<->right there
    (verifier: overlay the non-background cells of vmirror(I) onto I).

      * which region : the blank half-block of I (found by scanning I)
      * which axis   : horizontal mirror (the blank half is a COLUMN half of I)
      * how far      : block width w = wi // 2, measured from I's width
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # Background: the blank half-block makes bgc the majority colour of I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    w = wi // 2                       # half-block width (odd wi -> 1 spare col)
    left_c, right_c = 0, wi - w

    left_has = bool(np.any(I[:, left_c:left_c + w] != bgc))
    if left_has:
        src_c, dst_c = left_c, right_c
    else:
        src_c, dst_c = right_c, left_c

    src_block = I[:, src_c:src_c + w]

    # Full-rectangle selections: these ops act on the whole half-block
    # (background included), so a rectangle is exactly the intended cell set.
    src_rect = [(r, c) for r in range(hi) for c in range(src_c, src_c + w)]
    dst_rect = [(r, c) for r in range(hi) for c in range(dst_c, dst_c + w)]

    # 1. copy the content block out of the INPUT grid
    ops.append(28); sels.append(sel_of(src_rect))
    # 2. lay it down over the blank block (paste origin = top-left cell)
    ops.append(30); sels.append(sel_of([(0, dst_c)]))
    # 3. mirror that freshly pasted block left<->right in place
    if not np.array_equal(src_block, src_block[:, ::-1]):
        ops.append(26); sels.append(sel_of(dst_rect))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 496994bd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 496994bd"
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
                                f"for task 496994bd"
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
                    f"Failed to build a complete episode for task 496994bd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"496994bd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
