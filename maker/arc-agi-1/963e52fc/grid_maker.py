"""
ARC Task: 963e52fc (RE-ARC) — LLM-generated grid_maker
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

from dsl import *          # noqa: F401,F403  (RE-ARC DSL: unifint, interval, canvas, paint, shift, sfilter, lefthalf, ...)
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------- 1
def sample_colors(num_examples=None) -> dict:
    """Episode-level colours. The rule (horizontal-period tiling) is purely
    structural, so only the background needs to be pinned; the pattern palette
    is pinned too for visual consistency across the episode."""
    cols = list(range(10))
    bgc = random.choice(cols)
    ccols_pool = [c for c in cols if c != bgc]
    random.shuffle(ccols_pool)
    return {"bgc": bgc, "ccols_pool": ccols_pool}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols_pool=None) -> dict:
    if ccols_pool is None:
        ccols_pool = [c for c in range(10) if c != bgc]
        random.shuffle(ccols_pool)

    h_ub = max(3, min(30, max_h))
    w_ub = max(6, min(15, max_w // 2))          # output width is 2*w

    h = unifint(diff_lb, diff_ub, (3, h_ub))
    w = unifint(diff_lb, diff_ub, (6, w_ub))
    p = unifint(diff_lb, diff_ub, (2, max(2, w // 2)))

    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = list(ccols_pool[:numc])

    obj = set()
    for j in range(p):
        ub = unifint(diff_lb, diff_ub, (0, h // 2))
        ub = h // 2 - ub
        lb = unifint(diff_lb, diff_ub, (ub, h - 1))
        numcells = unifint(diff_lb, diff_ub, (1, lb - ub + 1))
        for ii in random.sample(list(range(ub, lb + 1)), numcells):
            obj.add((random.choice(ccols), (ii, j)))

    go = canvas(bgc, (h, w * 2))
    minobj = obj | shift(obj, (0, p))
    addonw = random.randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = random.randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    go = paint(go, fullobj)
    for j in range((2 * w) // (2 * p) + 1):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    gi = lefthalf(go)
    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O):
    """Rule (measured from I): the grid is horizontally periodic with minimal
    period p. Widen the canvas to double width and keep stamping the p-wide
    base slab of I at every multiple of p until the new canvas is covered."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # --- measure the horizontal period of I (dynamic, from I only) ---
    p = wi
    for q in range(1, wi):
        ok = True
        for c in range(wi - q):
            if not np.array_equal(I[:, c], I[:, c + q]):
                ok = False
                break
        if ok:
            p = q
            break

    ops, sels = [], []

    # 1) widen the canvas to the output size.
    #    ResizeGrid is a transparent copy: the left half keeps I exactly
    #    (its 0-valued cells stay 0), the new right half is all 0.
    #    bbox = the whole output rectangle, which is exactly the intended region.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2) copy the base period slab (columns 0..p-1 of the INPUT) to the clipboard.
    #    bbox = exactly that full rectangle.
    ops.append(28); sels.append([0, 0, hi - 1, p - 1])

    # 3) stamp the slab at every period offset that reaches past the old width.
    #    Paste is transparent, so cells whose base value is 0 correctly stay 0.
    c = (wi // p) * p
    while c < wo:
        ops.append(30); sels.append([0, c, 0, 0])   # paste origin only
        c += p

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 963e52fc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 963e52fc"
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
                                f"for task 963e52fc"
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
                    f"Failed to build a complete episode for task 963e52fc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"963e52fc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
