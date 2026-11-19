"""
ARC Task: b9b7f026 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; the rule (which rectangle has a
    # hole) does not depend on the foreground colours themselves.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    from random import randint, choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + int(round(lb * (b - a)))
        hi_ = a + int(round(ub * (b - a)))
        lo, hi_ = max(a, min(lo, b)), max(a, min(hi_, b))
        if hi_ < lo:
            hi_ = lo
        return randint(lo, hi_)

    if bgc is None:
        bgc = random.choice(list(range(10)))
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))

    while True:
        h = unifint(diff_lb, diff_ub, (10, H))
        w = unifint(diff_lb, diff_ub, (10, W))
        remcols = [c for c in range(10) if c != bgc]
        gi = [[bgc] * w for _ in range(h)]
        num = unifint(diff_lb, diff_ub, (1, 9))
        indss = {(i, j) for i in range(h) for j in range(w)}
        maxtrials = 4 * num
        succ, tr = 0, 0
        outcol = None
        while succ < num and tr <= maxtrials:
            if len(remcols) == 0 or len(indss) == 0:
                break
            oh = randint(3, 7)
            ow = randint(3, 7)
            subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
            if len(subs) == 0:
                tr += 1
                continue
            loci, locj = choice(subs)
            bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
            col = choice(remcols)
            if bd.issubset(indss):
                remcols = [c for c in remcols if c != col]
                for (i, j) in bd:
                    gi[i][j] = col
                succ += 1
                indss = indss - bd
                if outcol is None:
                    outcol = col
                    cands = [(i, j) for i in range(loci + 1, loci + oh - 1)
                             for j in range(locj + 1, locj + ow - 1)]
                    pts = sample(cands, 2) if len(cands) > 2 else cands
                    r0 = min(p[0] for p in pts); r1 = max(p[0] for p in pts)
                    c0 = min(p[1] for p in pts); c1 = max(p[1] for p in pts)
                    for i in range(r0, r1 + 1):
                        for j in range(c0, c1 + 1):
                            gi[i][j] = bgc
            tr += 1
        if outcol is None:
            continue
        return {"input": gi, "output": [[outcol]]}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Rectangles never reach the last row/col, so the bottom-right cell is background.
    bgc = int(I[hi - 1, wi - 1])

    # Reading the task row by row: an intact rectangle of colour d puts the same
    # number of d-cells (its width) in every row it touches, so its per-row counts
    # take just two values {0, width}. In the punched rectangle, the rows that cross
    # the hole hold fewer cells, which gives a third count. The first such
    # short row is where the hole shows up, so we take a cell of that colour there.
    target = None
    for d in sorted(set(I.flatten().tolist())):
        if d == bgc:
            continue
        counts = [(r, int((I[r] == d).sum())) for r in range(hi)]
        if len({n for _, n in counts}) > 2:
            full = max(n for _, n in counts)
            for r, n in counts:
                if 0 < n < full:
                    c = int(np.nonzero(I[r] == d)[0][0])  # wall cell left of the hole
                    target = (r, c)
                    break
            break

    if target is None:  # degenerate fallback: no punched rectangle
        rs, cs = np.nonzero(I != bgc)
        target = (int(rs[0]), int(cs[0])) if len(rs) else (0, 0)

    ops, sels = [], []
    # Crop the canvas to that one wall cell: the result is a 1x1 grid of the colour.
    ops.append(33); sels.append(sel_of([target]))
    ops.append(34); sels.append(sel_of([(0, 0)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task b9b7f026"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b9b7f026"
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
                                f"for task b9b7f026"
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
                    f"Failed to build a complete episode for task b9b7f026 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b9b7f026-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
