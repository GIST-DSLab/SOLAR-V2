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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    cols = list(range(10))
    nh = _unifint(diff_lb, diff_ub, (2, 6))
    nw = _unifint(diff_lb, diff_ub, (2, 6))
    nh = max(2, min(nh, (max_h + 1) // 3))
    nw = max(2, min(nw, (max_w + 1) // 3))
    remcols = [c for c in cols if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (3, 9))
    ccols = random.sample(remcols, ncols)

    def rand_go():
        return [[random.choice(ccols) for _ in range(nw)] for _ in range(nh)]
    go = rand_go()
    while len(set(tuple(r) for r in go)) < nh or \
            len(set(tuple(go[i][j] for i in range(nh)) for j in range(nw))) < nw:
        go = rand_go()

    h = _unifint(diff_lb, diff_ub, (3 * nh - 1, max_h))
    w = _unifint(diff_lb, diff_ub, (3 * nw - 1, max_w))
    hdist = [2] * nh
    for _ in range(h - 3 * nh + 1):
        hdist[random.randint(0, nh - 1)] += 1
    wdist = [2] * nw
    for _ in range(w - 3 * nw + 1):
        wdist[random.randint(0, nw - 1)] += 1

    rstarts, r = [], 0
    for k in range(nh):
        rstarts.append(r); r += hdist[k] + 1
    cstarts, c = [], 0
    for k in range(nw):
        cstarts.append(c); c += wdist[k] + 1

    for _attempt in range(200):
        gi = np.full((h, w), bgc, dtype=int)
        for i in range(nh):
            for j in range(nw):
                r0, c0 = rstarts[i], cstarts[j]
                gi[r0:r0 + hdist[i], c0:c0 + wdist[j]] = go[i][j]
                cells = [(rr, cc) for rr in range(r0, r0 + hdist[i])
                         for cc in range(c0, c0 + wdist[j])]
                k = _unifint(diff_lb, diff_ub, (1, len(cells) // 2))
                for (rr, cc) in random.sample(cells, k):
                    gi[rr, cc] = bgc
        sep_r = set(rstarts[i] + hdist[i] for i in range(nh - 1))
        sep_c = set(cstarts[j] + wdist[j] for j in range(nw - 1))
        ok = all(not np.all(gi[rr, :] == bgc) for rr in range(h) if rr not in sep_r) and \
             all(not np.all(gi[:, cc] == bgc) for cc in range(w) if cc not in sep_c)
        if ok:
            break
    else:
        raise ValueError("could not build unambiguous grid")
    return {"input": gi.tolist(), "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background = the colour that forms full separator rows and columns
    bgc = None
    for c in range(10):
        if any(np.all(I[r, :] == c) for r in range(hi)) and \
           any(np.all(I[:, k] == c) for k in range(wi)):
            bgc = c
            break
    sep_r = [np.all(I[r, :] == bgc) for r in range(hi)]
    sep_c = [np.all(I[:, k] == bgc) for k in range(wi)]

    def runs(flags):
        out, start = [], None
        for idx, f in enumerate(flags):
            if not f and start is None:
                start = idx
            if f and start is not None:
                out.append((start, idx)); start = None
        if start is not None:
            out.append((start, len(flags)))
        return out
    rbands, cbands = runs(sep_r), runs(sep_c)
    nh, nw = len(rbands), len(cbands)
    colors = [[None] * nw for _ in range(nh)]
    for i, (r0, r1) in enumerate(rbands):
        for j, (c0, c1) in enumerate(cbands):
            vals = [v for v in I[r0:r1, c0:c1].flatten().tolist() if v != bgc]
            colors[i][j] = max(set(vals), key=vals.count)

    ops, sels = [], []
    # shrink canvas to one cell per block
    ops.append(33); sels.append([0, 0, nh - 1, nw - 1])
    cur = I[:nh, :nw].copy()
    order = []
    for i in range(nh):
        for j in range(nw):
            if colors[i][j] not in order:
                order.append(colors[i][j])
    for col in order:
        cells = [(i, j) for i in range(nh) for j in range(nw)
                 if colors[i][j] == col and cur[i, j] != col]
        if cells:
            ops.append(int(col)); sels.append(sel_of(cells))
            for (i, j) in cells:
                cur[i, j] = col
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
