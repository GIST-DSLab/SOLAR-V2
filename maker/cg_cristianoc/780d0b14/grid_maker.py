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

# Both readings agree. The maker and the code-golf entry carve the task the same way.
# Uniform background rows and columns cut the grid into an nh x nw lattice of blocks.
# Each block is one unit and stands for its single non-background colour.
# The output is that colour lattice, read row band by row band.
# The golf entry takes "the first non-zero pixel per block". That is the same quantity
# (the block's only foreground colour), with bgc assumed to be 0.
# So the unit, the order and the fixed quantity do not change. The route below is the
# maker's route: write each block's colour into the corner cell it collapses to, then
# crop to the lattice.


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lo, hi):
        lo2 = lo + diff_lb * (hi - lo)
        hi2 = lo + diff_ub * (hi - lo)
        return random.randint(int(round(lo2)), max(int(round(lo2)), int(round(hi2))))

    if bgc is None:
        bgc = random.choice(range(10))
    max_h = min(max_h, 30)
    max_w = min(max_w, 30)
    nh_hi = max(2, min(6, (max_h + 1) // 3))
    nw_hi = max(2, min(6, (max_w + 1) // 3))
    nh = unifint(2, nh_hi)
    nw = unifint(2, nw_hi)
    remcols = [c for c in range(10) if c != bgc]
    ncols = unifint(3, 9)
    ccols = random.sample(remcols, ncols)

    while True:
        go = [[random.choice(ccols) for _ in range(nw)] for _ in range(nh)]
        rows = set(tuple(r) for r in go)
        colsets = set(tuple(go[i][j] for i in range(nh)) for j in range(nw))
        if len(rows) == nh and len(colsets) == nw:
            break

    hmin, wmin = 3 * nh - 1, 3 * nw - 1
    h = unifint(hmin, max(hmin, max_h))
    w = unifint(wmin, max(wmin, max_w))
    hdist = [2] * nh
    for _ in range(h - hmin):
        hdist[random.randint(0, nh - 1)] += 1
    wdist = [2] * nw
    for _ in range(w - wmin):
        wdist[random.randint(0, nw - 1)] += 1

    for _attempt in range(100):
        gi = np.full((h, w), bgc, dtype=int)
        r = 0
        rb = []
        for i in range(nh):
            rb.append((r, r + hdist[i]))
            r += hdist[i] + 1
        c = 0
        cb = []
        for j in range(nw):
            cb.append((c, c + wdist[j]))
            c += wdist[j] + 1
        for i, (r0, r1) in enumerate(rb):
            for j, (c0, c1) in enumerate(cb):
                gi[r0:r1, c0:c1] = go[i][j]
        # sparsify each same-colour connected component (4-connectivity)
        seen = np.zeros_like(gi, dtype=bool)
        base = gi.copy()
        for sr in range(h):
            for sc in range(w):
                if seen[sr, sc] or base[sr, sc] == bgc:
                    continue
                col = base[sr, sc]
                comp = []
                stack = [(sr, sc)]
                seen[sr, sc] = True
                while stack:
                    a, b = stack.pop()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < h and 0 <= y < w and not seen[x, y] and base[x, y] == col:
                            seen[x, y] = True
                            stack.append((x, y))
                k = unifint(1, max(1, len(comp) // 2))
                for (a, b) in random.sample(comp, k):
                    gi[a, b] = bgc
        # validity: exactly nh-1 / nw-1 uniform lines, each block keeps a fg cell
        urows = [i for i in range(h) if len(set(gi[i].tolist())) == 1]
        ucols = [j for j in range(w) if len(set(gi[:, j].tolist())) == 1]
        if len(urows) == nh - 1 and len(ucols) == nw - 1:
            ok = all((gi[r0:r1, c0:c1] != bgc).any() for (r0, r1) in rb for (c0, c1) in cb)
            if ok:
                return {"input": gi.tolist(), "output": [list(rw) for rw in go]}
    raise ValueError("failed to generate")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # 1. The frontiers are the uniform rows and columns. Their colour is the background.
    frow = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if frow:
        bgc = int(I[frow[0], 0])
    else:
        bgc = int(I[0, fcol[0]])

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
    nh, nw = len(rbands), len(cbands)

    ops, sels = [], []
    # 2. Read each block's single non-background colour from I, row band by row band.
    #    Write it into cell (i, j) of the top-left corner, the cell this block collapses to.
    for i, (r0, r1) in enumerate(rbands):
        for j, (c0, c1) in enumerate(cbands):
            blk = I[r0:r1 + 1, c0:c1 + 1]
            fgs = set(blk.flatten().tolist()) - {bgc}
            fg = int(next(iter(fgs)))
            if int(I[i, j]) != fg:          # this cell already holds the colour, so no op
                ops.append(fg)
                sels.append(sel_of([(i, j)]))

    # 3. Keep only the nh x nw lattice. The selection is the full corner rectangle.
    ops.append(33)
    sels.append([0, 0, nh - 1, nw - 1])
    ops.append(34)
    sels.append([0, 0, nh - 1, nw - 1])
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
