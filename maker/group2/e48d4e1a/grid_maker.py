"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # cross colour kept non-zero so each bar is a grabbable (non-zero) object for Move
    fgc = random.choice([c for c in cols if c != bgc and c != 0])
    dotc = random.choice([c for c in cols if c not in (bgc, fgc)])
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc}


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=1, dotc=2) -> dict:
    k = random.choice((0, 1, 2, 3))  # quarter turns applied to both grids
    if k % 2 == 0:
        H, W = max_h, max_w
    else:
        H, W = max_w, max_h
    h = _unifint(diff_lb, diff_ub, (3, max(3, H)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, W)))
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    inds = [(i, j) for i in range(loci) for j in range(locj)]
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = _unifint(diff_lb, diff_ub, (1, maxn))
    ss = random.sample(inds, nn)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci, :] = fgc
    gi[:, locj] = fgc
    for (i, j) in ss:
        gi[i, j] = dotc
    go[loci + nn, :] = fgc
    go[:, locj + nn] = fgc
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # The cross: the one full uniform row and the one full uniform column of the same colour.
    hrow = vcol = None
    fgc = None
    for r in range(hi):
        if np.all(I[r, :] == I[r, 0]):
            cand = int(I[r, 0])
            cs = [c for c in range(wi) if np.all(I[:, c] == cand)]
            if cs:
                hrow, vcol, fgc = r, cs[0], cand
                break
    # background = most common colour off the cross; dots = the remaining colour
    rest = [int(v) for v in I.flatten() if v != fgc]
    cnt = Counter(rest)
    bgc = cnt.most_common(1)[0][0]
    dotc = [c for c in cnt if c != bgc][0]
    dots = [(int(r), int(c)) for r, c in np.argwhere(I == dotc)]
    nn = len(dots)
    mean_r = sum(r for r, _ in dots) / nn
    mean_c = sum(c for _, c in dots) / nn
    di = sign(hrow - mean_r)   # cross slides away from the dots, nn cells on each axis
    dj = sign(vcol - mean_c)
    new_hrow = hrow + di * nn
    new_vcol = vcol + dj * nn

    ops, sels = [], []
    # 1. consume the counter dots: all cells of the dot colour -> background
    ops.append(bgc); sels.append(sel_of(dots))

    if fgc == 0:
        # zero-coloured bars cannot be grabbed by Move (object buffer ignores 0): paint bars
        ops.append(bgc); sels.append(sel_of([(hrow, c) for c in range(wi)]))
        ops.append(bgc); sels.append(sel_of([(r, vcol) for r in range(hi)]))
        ops.append(fgc); sels.append(sel_of([(new_hrow, c) for c in range(wi)]))
        ops.append(fgc); sels.append(sel_of([(r, new_vcol) for r in range(hi)]))
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # 2. slide the horizontal bar vertically by nn (grab once, then empty selections)
    row_cells = [(hrow, c) for c in range(wi)]
    mop = 21 if di > 0 else 20
    ops.append(mop); sels.append(sel_of(row_cells))
    for _ in range(nn - 1):
        ops.append(mop); sels.append(sel_of([]))
    # repair the bar's old footprint: background, except the vertical bar's own cell
    if bgc != 0:
        ops.append(bgc); sels.append(sel_of([(hrow, c) for c in range(wi) if c != vcol]))
    ops.append(fgc); sels.append(sel_of([(hrow, vcol)]))

    # 3. slide the vertical bar horizontally by nn
    col_cells = [(r, vcol) for r in range(hi)]
    mop = 22 if dj > 0 else 23
    ops.append(mop); sels.append(sel_of(col_cells))
    for _ in range(nn - 1):
        ops.append(mop); sels.append(sel_of([]))
    # repair its old footprint: background, except where the moved horizontal bar crosses it
    if bgc != 0:
        ops.append(bgc); sels.append(sel_of([(r, vcol) for r in range(hi) if r != new_hrow]))
    ops.append(fgc); sels.append(sel_of([(new_hrow, vcol)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
