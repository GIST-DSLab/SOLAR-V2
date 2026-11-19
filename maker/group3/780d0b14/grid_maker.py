"""
ARC Task: 780d0b14 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    return random.randint(min(lo, hi), max(lo, hi))


def _bands(n, fl):
    res, start = [], None
    for i in range(n):
        if i in fl:
            if start is not None:
                res.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        res.append((start, n - 1))
    return res


def sample_colors(num_examples=None) -> dict:
    bgc = random.randint(0, 9)
    rem = [c for c in range(10) if c != bgc]
    ccols = random.sample(rem, random.randint(3, 9))
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols=None) -> dict:
    if ccols is None:
        ccols = random.sample([c for c in range(10) if c != bgc], 3)
    nh = min(_unifint(diff_lb, diff_ub, (2, 6)), (max_h + 1) // 3)
    nw = min(_unifint(diff_lb, diff_ub, (2, 6)), (max_w + 1) // 3)
    while True:
        go = [[random.choice(ccols) for _ in range(nw)] for _ in range(nh)]
        rows_ok = all(go[i] != go[i + 1] for i in range(nh - 1))
        cols_ok = all([go[r][j] for r in range(nh)] != [go[r][j + 1] for r in range(nh)]
                      for j in range(nw - 1))
        if rows_ok and cols_ok:
            break
    h = _unifint(diff_lb, diff_ub, (3 * nh - 1, max_h))
    w = _unifint(diff_lb, diff_ub, (3 * nw - 1, max_w))
    hdist = [2] * nh
    for _ in range(h - 3 * nh + 1):
        hdist[random.randint(0, nh - 1)] += 1
    wdist = [2] * nw
    for _ in range(w - 3 * nw + 1):
        wdist[random.randint(0, nw - 1)] += 1
    while True:
        gi = np.full((h, w), bgc, dtype=int)
        r = 0
        for i in range(nh):
            c = 0
            for j in range(nw):
                blk = [(rr, cc) for rr in range(r, r + hdist[i]) for cc in range(c, c + wdist[j])]
                for rr, cc in blk:
                    gi[rr, cc] = go[i][j]
                k = _unifint(diff_lb, diff_ub, (1, len(blk) // 2))
                for rr, cc in random.sample(blk, k):
                    gi[rr, cc] = bgc
                c += wdist[j] + 1
            r += hdist[i] + 1
        # reject inputs where holes create a fake all-bgc frontier line
        frow = [x for x in range(h) if len(set(gi[x].tolist())) == 1]
        fcol = [x for x in range(w) if len(set(gi[:, x].tolist())) == 1]
        if len(_bands(h, set(frow))) == nh and len(_bands(w, set(fcol))) == nw:
            break
    return {"input": gi.tolist(), "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # frontier lines: uniform rows / columns of the background colour
    frow = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    rb = _bands(hi, set(frow))
    cb = _bands(wi, set(fcol))
    bgc = int(I[frow[0], 0])
    nh, nw = len(rb), len(cb)

    # colour carried by each block (row-band i x col-band j), read from I
    blockcol = {}
    for i, (r0, r1) in enumerate(rb):
        for j, (c0, c1) in enumerate(cb):
            vals = set(I[r0:r1 + 1, c0:c1 + 1].flatten().tolist()) - {bgc}
            blockcol[(i, j)] = int(next(iter(vals)))

    ops, sels = [], []
    # 1. shrink the canvas to one cell per block (transparent crop of the top-left corner)
    ops.append(33); sels.append([0, 0, nh - 1, nw - 1])
    grid = np.zeros((nh, nw), dtype=int)
    for i in range(nh):
        for j in range(nw):
            grid[i, j] = I[i, j]          # zeros stay zero after the transparent crop

    # 2. one colour at a time: paint every block-cell whose block carries that colour
    #    (skip cells already holding it in the current grid)
    for col in sorted(set(blockcol.values())):
        cells = [(i, j) for (i, j), v in sorted(blockcol.items())
                 if v == col and grid[i, j] != col]
        if cells:
            ops.append(col); sels.append(sel_of(cells))
            for i, j in cells:
                grid[i, j] = col

    ops.append(34); sels.append([0, 0, nh - 1, nw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
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
                                f"for task 780d0b14"
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
                    f"Failed to build a complete episode for task 780d0b14 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"780d0b14-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
