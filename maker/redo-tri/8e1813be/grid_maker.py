"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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

# Discrete structural variant: the generator may dmirror the whole instance,
# turning the row-bars into column-bars (and the striped output into columns).
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test variant was demonstrated

    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, mirrored=None) -> dict:
    def _unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if mirrored is None:
        mirrored = random.choice([True, False])

    cols = [c for c in range(10) if c != bgc and c != sqc]

    # Before the optional dmirror the grid is h2 rows x w cols.
    hlim = min(30, max_w if mirrored else max_h)   # bound on h2
    wlim = min(30, max_h if mirrored else max_w)   # bound on w

    nbars_ub = min(8, wlim - 3, hlim // 3)
    nbars_ub = max(3, nbars_ub)
    nbars = _unifint(diff_lb, diff_ub, (3, nbars_ub))
    nbars = min(nbars, len(cols))

    w = _unifint(diff_lb, diff_ub, (nbars + 3, max(nbars + 3, wlim)))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, hlim - nbars)))

    ccols = random.sample(cols, nbars)
    gi = [[c] * w for c in ccols]          # the bars, one per row
    go = [[c] * nbars for c in ccols]      # striped answer canvas

    for _ in range(hmarg):
        idx = random.randint(0, nbars - 1)
        gi.insert(idx, [bgc] * w)          # background margin rows

    h2 = nbars + hmarg
    loci = random.randint(1, h2 - nbars - 2)
    locj = random.randint(1, w - nbars - 2)

    for i in range(loci, loci + nbars):
        for j in range(locj, locj + nbars):
            gi[i][j] = sqc
    for i in range(loci - 1, loci + nbars + 1):
        for j in range(locj - 1, locj + nbars + 1):
            if i in (loci - 1, loci + nbars) or j in (locj - 1, locj + nbars):
                if 0 <= i < h2 and 0 <= j < w:
                    gi[i][j] = bgc

    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    # ---- measure the rule entirely from I -------------------------------
    # Every colour in I is one of: a bar (all its cells lie in a single row,
    # or a single column), the marker square (a solid k x k block), or the
    # background (spread over many rows AND columns, and not solid).
    hbars, vbars, square = [], [], None
    for c in sorted(set(I.flatten().tolist())):
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh == 1:
            hbars.append((r0, int(c)))
        elif bw == 1:
            vbars.append((c0, int(c)))
        elif bh == bw and bh > 1 and len(pts) == bh * bw:
            square = (r0, c0, bh)

    horizontal = len(hbars) >= len(vbars)
    bars = sorted(hbars) if horizontal else sorted(vbars)
    seq = [col for _, col in bars]          # bar colours, uppermost / leftmost first
    n = len(seq)                            # canvas side = number of bars

    # the marker square is where the striped answer gets drawn
    if square is not None:
        sr, sc = square[0], square[1]
    else:
        sr, sc = 0, 0
    sr = max(0, min(sr, hi - n))
    sc = max(0, min(sc, wi - n))

    # ---- draw one stripe per bar, in bar order, onto the square ---------
    for k, col in enumerate(seq):
        if horizontal:
            cells = [(sr + k, sc + j) for j in range(n)]      # k-th row of the square
        else:
            cells = [(sr + i, sc + k) for i in range(n)]      # k-th column of the square
        ops.append(int(col))
        sels.append(sel_of(cells))

    # ---- keep only the square: selection IS exactly that full rectangle --
    ops.append(33)
    sels.append(sel_of([(sr + i, sc + j) for i in range(n) for j in range(n)]))

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])   # whole (square) result grid
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
