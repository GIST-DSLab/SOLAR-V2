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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour role; the pattern colours
    # are just whatever the periodic stripe contains and the rule ignores them.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    w_ub = max(6, min(15, max_w // 2))
    w = unifint(diff_lb, diff_ub, (6, w_ub))
    p = unifint(diff_lb, diff_ub, (2, w // 2))
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numc)
    obj = set()
    for j in range(p):
        ub = unifint(diff_lb, diff_ub, (0, h // 2))
        ub = h // 2 - ub
        lb = unifint(diff_lb, diff_ub, (ub, h - 1))
        numcells = unifint(diff_lb, diff_ub, (1, lb - ub + 1))
        for ii in random.sample(range(ub, lb + 1), numcells):
            obj.add((random.choice(ccols), (ii, j)))
    W = w * 2
    go = [[bgc] * W for _ in range(h)]

    def shift(o, dj):
        return {(c, (i, j + dj)) for c, (i, j) in o}

    minobj = obj | shift(obj, p)
    addonw = random.randint(0, p)
    addon = {cell for cell in obj if cell[1][1] < addonw}
    fullobj = minobj | addon
    leftshift = random.randint(0, addonw)
    fullobj = shift(fullobj, -leftshift)

    def paint(o):
        for c, (i, j) in o:
            if 0 <= i < h and 0 <= j < W:
                go[i][j] = c

    paint(fullobj)
    for j in range((2 * w) // (2 * p) + 1):
        paint(shift(fullobj, j * 2 * p))
    gi = [row[:w] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (golf-style): the unit is not the single minimal period but the
    widest block of WHOLE periods that sits at the left of I (width q = k*p,
    the largest multiple of p that fits in I's width, like the golf's
    6-column chunk). Widen the canvas to double width, then keep stamping that
    multi-period block end to end (at q, 2q, ...) until the new canvas is covered."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi, 2 * wi  # output = input doubled in width (rule, not read from O)

    # minimal horizontal period of I (measured from I only)
    p = wi
    for q in range(1, wi):
        if all(np.array_equal(I[:, c], I[:, c + q]) for c in range(wi - q)):
            p = q
            break
    blk = (wi // p) * p  # widest block of whole periods inside I

    ops, sels = [], []
    # 1) widen canvas to double width; bbox = whole new canvas (intended full rectangle).
    #    Transparent copy keeps I in the left half; the new right half is 0.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # 2) copy the multi-period block (columns 0..blk-1 of the INPUT); full rectangle.
    ops.append(28); sels.append([0, 0, hi - 1, blk - 1])
    # 3) stamp the block end to end, starting where the block ends, until covered.
    c = blk
    while c < wo:
        ops.append(30); sels.append([0, c, 0, 0])  # paste origin only
        c += blk

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
