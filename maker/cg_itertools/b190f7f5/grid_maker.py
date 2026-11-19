"""
ARC Task: b190f7f5 (RE-ARC) — LLM-generated grid_maker
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

# The code-golf entry reads the task the same way the maker does. The two input
# halves are a single-colour "mask" grid and a multi-colour "palette" grid. The
# output is their Kronecker product: block (a, b) holds the mask's cell layout,
# painted with palette[a][b]. The golf entry transposes wide grids and swaps the
# halves, which only normalises orientation. Both readings use the same unit (one
# stamp of the mask per coloured palette cell) and the same order, so the
# operations are unchanged. What does change is the split detection. Choosing the
# axis by hi > wi fails on 5x4 or 4x5 inputs, so it now copies the verifier's scoring.


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    srccol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "srccol": srccol}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, srccol=None, **kw) -> dict:
    from math import isqrt
    fullcols = list(range(10))
    if bgc is None:
        bgc = random.choice(fullcols)
    cols = [c for c in fullcols if c != bgc]
    if srccol is None or srccol == bgc:
        srccol = random.choice(cols)

    def unifint(lb, ub):
        lo, hi_ = lb, ub
        if hi_ <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi_, round(lo + d * (hi_ - lo) + random.uniform(-0.5, 0.5))))

    hcap = max(2, min(5, isqrt(max_h)))
    wcap = max(2, min(5, isqrt(max_w)))
    h = unifint(2, hcap)
    w = unifint(2, wcap)
    inds = [(i, j) for i in range(h) for j in range(w)]

    numcd = unifint(0, (h * w) // 2)
    numc = random.choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    numcd2 = unifint(0, (h * w) // 2)
    numc2 = random.choice((numcd2, h * w - numcd2))
    numc2 = min(max(2, numc2), h * w - 1)

    srclocs = random.sample(inds, numc)
    remcols = [c for c in cols if c != srccol]
    numcols = unifint(2, 8)
    trglocs = random.sample(inds, numc2)
    ccols = random.sample(remcols, numcols)
    fixc1 = random.choice(ccols)
    trgcols = [fixc1, random.choice([c for c in ccols if c != fixc1])] + \
              [random.choice(ccols) for _ in trglocs[2:]]

    gisrc = [[bgc] * w for _ in range(h)]
    for (i, j) in srclocs:
        gisrc[i][j] = srccol
    gitrg = [[bgc] * w for _ in range(h)]
    for col, (i, j) in zip(trgcols, trglocs):
        gitrg[i][j] = col

    pair = random.choice(([gisrc, gitrg], [gitrg, gisrc]))
    can_h = 2 * w <= max_w
    can_v = 2 * h <= max_h
    if can_h and (not can_v or random.random() < 0.5):
        gi = [pair[0][r] + pair[1][r] for r in range(h)]
    else:
        gi = [row[:] for row in pair[0]] + [row[:] for row in pair[1]]

    go = [[bgc] * (w * w) for _ in range(h * h)]
    for (a, b) in trglocs:
        col = gitrg[a][b]
        for (i, j) in srclocs:
            go[a * h + i][b * w + j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def ncol(g):
        return len(set(g.flatten().tolist()))

    # candidate splits (both halves must be equal-shaped)
    cands = []
    if hi % 2 == 0:
        cands.append((I[:hi // 2, :], I[hi // 2:, :]))
    if wi % 2 == 0:
        cands.append((I[:, :wi // 2], I[:, wi // 2:]))

    def score(pr):
        a, b = pr
        s = 0
        if 2 in (ncol(a), ncol(b)):
            s += 1
        pa, pb = set(a.flatten().tolist()), set(b.flatten().tolist())
        if len(pa & pb) == 1:
            s += 1
        if 1 not in (min(a.shape), min(b.shape)):
            s += 1
        if min(ncol(a), ncol(b)) == 2 and max(ncol(a), ncol(b)) >= 3:
            s += 1
        return s

    halves = max(cands, key=score)
    if ncol(halves[0]) <= ncol(halves[1]):
        source, target = halves
    else:
        target, source = halves

    common = set(source.flatten().tolist()) & set(target.flatten().tolist())
    bgc = int(sorted(common)[0])

    hs, ws = source.shape
    Hs, Ws = hs * hs, ws * ws
    stamp = [(r, c) for r in range(hs) for c in range(ws) if int(source[r, c]) != bgc]

    # resize canvas to the Kronecker-product size; whole new canvas (a full rectangle)
    ops.append(33); sels.append([0, 0, Hs - 1, Ws - 1])
    # lay a uniform background base over the whole canvas (clears leftover input cells)
    ops.append(bgc); sels.append([0, 0, Hs - 1, Ws - 1])

    # one stamp of the source mask per coloured target cell, in that cell's colour
    for r in range(hs):
        for c in range(ws):
            col = int(target[r, c])
            if col == bgc:
                continue
            cells = [(r * hs + sr, c * ws + sc) for (sr, sc) in stamp]
            if cells:
                ops.append(col); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, Hs - 1, Ws - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
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
                                f"for task b190f7f5"
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
                    f"Failed to build a complete episode for task b190f7f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b190f7f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
