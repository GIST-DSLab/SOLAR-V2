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


def sample_colors(num_examples=None) -> dict:
    # Only the background/frontier colour matters for the rule
    # ("each block -> its non-background colour"); block colours are free.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def uf(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    nh_ub = max(2, min(6, (max_h + 1) // 3))
    nw_ub = max(2, min(6, (max_w + 1) // 3))
    nh = uf(diff_lb, diff_ub, (2, nh_ub))
    nw = uf(diff_lb, diff_ub, (2, nw_ub))

    remcols = [c for c in range(10) if c != bgc]
    ncols = uf(diff_lb, diff_ub, (3, 9))
    ccols = random.sample(remcols, ncols)

    # output = block-colour grid; all rows distinct and all cols distinct
    while True:
        go = [[random.choice(ccols) for _ in range(nw)] for _ in range(nh)]
        rows_uniq = len({tuple(r) for r in go}) == nh
        cols_uniq = len({tuple(go[i][j] for i in range(nh)) for j in range(nw)}) == nw
        if rows_uniq and cols_uniq:
            break

    h = uf(diff_lb, diff_ub, (3 * nh - 1, max_h))
    w = uf(diff_lb, diff_ub, (3 * nw - 1, max_w))

    hdist = [2 for _ in range(nh)]
    for _ in range(h - 3 * nh + 1):
        hdist[random.randint(0, nh - 1)] += 1
    wdist = [2 for _ in range(nw)]
    for _ in range(w - 3 * nw + 1):
        wdist[random.randint(0, nw - 1)] += 1

    def expand_row(vals):
        row = []
        for j, v in enumerate(vals):
            row += [v] * wdist[j]
            if j < nw - 1:
                row.append(bgc)
        return row

    base = []
    for k in range(nh):
        er = expand_row(go[k])
        for _ in range(hdist[k]):
            base.append(list(er))
        if k < nh - 1:
            base.append([bgc] * w)

    # band ranges and the bgc frontier lines that separate them
    rr, s = [], 0
    frs = []
    for k in range(nh):
        rr.append((s, s + hdist[k] - 1))
        s += hdist[k]
        if k < nh - 1:
            frs.append(s)
            s += 1
    cc, s = [], 0
    fcs = []
    for j in range(nw):
        cc.append((s, s + wdist[j] - 1))
        s += wdist[j]
        if j < nw - 1:
            fcs.append(s)
            s += 1

    # punch bgc holes into each block (at most half of it)
    for _ in range(50):
        gi = [list(r) for r in base]
        for k in range(nh):
            for j in range(nw):
                cells = [(r, c)
                         for r in range(rr[k][0], rr[k][1] + 1)
                         for c in range(cc[j][0], cc[j][1] + 1)]
                n = uf(diff_lb, diff_ub, (1, len(cells) // 2))
                for (r, c) in random.sample(cells, n):
                    gi[r][c] = bgc
        # the only uniform lines must be the real frontiers
        ur = {i for i in range(h) if len(set(gi[i])) == 1}
        uc = {j for j in range(w) if len({gi[i][j] for i in range(h)}) == 1}
        if ur == set(frs) and uc == set(fcs):
            return {"input": tuple(tuple(r) for r in gi),
                    "output": tuple(tuple(r) for r in go)}
    raise ValueError("could not build unambiguous frontiers")


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # 1. the grid is cut by uniform bgc lines (frontiers)
    frow = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    bgc = int(I[frow[0], 0])

    def bands(n, fl):
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

    rbands = bands(hi, set(frow))
    cbands = bands(wi, set(fcol))

    ops, sels = [], []

    # 2. every block (row-band i x col-band j) holds exactly one non-bgc colour;
    #    collapse that whole block onto cell (i, j) of the top-left corner.
    for i, (r0, r1) in enumerate(rbands):
        for j, (c0, c1) in enumerate(cbands):
            blk = I[r0:r1 + 1, c0:c1 + 1]
            fg = int(next(iter(set(blk.flatten().tolist()) - {bgc})))
            if int(I[i, j]) != fg:          # cell already holds it -> no op
                ops.append(fg)
                sels.append([i, j, 0, 0])

    # 3. keep only the block grid
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
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
