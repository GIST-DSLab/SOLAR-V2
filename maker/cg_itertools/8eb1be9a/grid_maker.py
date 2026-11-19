"""
ARC Task: 8eb1be9a (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 9)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, ccols=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if bgc is None:
        bgc = random.randint(0, 9)
    if not ccols:
        rem = [c for c in range(10) if c != bgc]
        ccols = random.sample(rem, random.randint(1, 9))
    ccols = [c for c in ccols if c != bgc] or [(bgc + 1) % 10]

    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    oh = unifint(diff_lb, diff_ub, (2, max(2, h // 3)))
    ow = unifint(diff_lb, diff_ub, (2, w))
    bounds = [(i, j) for i in range(oh) for j in range(ow)]
    ncells = unifint(diff_lb, diff_ub, (2, max(2, (oh * ow) // 3 * 2)))
    cells = random.sample(bounds, ncells)
    mi = min(i for i, _ in cells)
    mj = min(j for _, j in cells)
    cells = [(i - mi, j - mj) for i, j in cells]
    oh = max(i for i, _ in cells) + 1
    ow = max(j for _, j in cells) + 1
    obj = [(random.choice(ccols), (i, j)) for i, j in cells]
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    obj = [(c, (i + loci, j + locj)) for c, (i, j) in obj]

    gi = [[bgc] * w for _ in range(h)]
    for c, (i, j) in obj:
        gi[i][j] = c
    go = [row[:] for row in gi]
    for k in range(h // oh + 1):
        for d in (-oh * k, oh * k):
            for c, (i, j) in obj:
                ii = i + d
                if 0 <= ii < h:
                    go[ii][j] = c
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background: the canvas colour (object covers a small minority of cells).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not cells:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1

    # Reading: every row of the object band is its own unit; output row t takes
    # the object row with the same phase (t mod oh). Propagate each object row
    # up and down the grid with period oh, nearest copies first.
    for r in range(r0, r1 + 1):
        row_cells = [(r, c) for c in range(c0, c1 + 1) if I[r, c] != bgc]
        if not row_cells:
            continue
        nonzero = any(I[rr, cc] != 0 for rr, cc in row_cells)
        zeros = [c for rr, c in row_cells if I[rr, c] == 0]

        targets = []
        k = 1
        while True:
            up, dn = r - k * oh, r + k * oh
            if up < 0 and dn >= h:
                break
            if up >= 0:
                targets.append(up)
            if dn < h:
                targets.append(dn)
            k += 1

        if not targets:
            continue
        if nonzero:
            # exact rectangle: this object row's segment across the band's width
            ops.append(28); sels.append([r, c0, 0, ow - 1])
        for t in targets:
            if nonzero:
                ops.append(30); sels.append([t, c0, 0, 0])
            if zeros:
                ops.append(0); sels.append(sel_of([(t, c) for c in zeros]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
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
                                f"for task 8eb1be9a"
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
                    f"Failed to build a complete episode for task 8eb1be9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8eb1be9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
