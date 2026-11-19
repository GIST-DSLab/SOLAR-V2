"""
ARC Task: 0962bcdd (RE-ARC) — LLM-generated grid_maker
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
    # Colours 3 and 4 are never used. The rule doesn't depend on which colours the
    # crosses have, so only the background is fixed for the episode.
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))

    def connect(a, b):
        (i0, j0), (i1, j1) = a, b
        n = max(abs(i1 - i0), abs(j1 - j0))
        di = (i1 > i0) - (i1 < i0)
        dj = (j1 > j0) - (j1 < j0)
        return {(i0 + k * di, j0 + k * dj) for k in range(n + 1)}

    cols = [c for c in range(10) if c not in (3, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (2, 7))
    ccols = random.sample(remcols, numc)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    subs = [(i, j) for i in range(h) for j in range(w) if i < h - 5 and j < w - 5]
    while succ < num and tr <= maxtrials:
        if not subs:
            break
        loci, locj = random.choice(subs)
        bd = {(i, j) for i in range(loci, loci + 5) for j in range(locj, locj + 5)}
        if bd.issubset(indss):
            ca, cb = random.sample(ccols, 2)
            cp = (loci + 2, locj + 2)
            lins12 = connect((loci, locj), (loci + 4, locj + 4)) | connect((loci + 4, locj), (loci, locj + 4))
            lins34 = connect((loci + 2, locj), (loci + 2, locj + 4)) | connect((loci, locj + 2), (loci + 4, locj + 2))
            for (i, j) in lins34:
                go[i, j] = cb
            for (i, j) in lins12:
                go[i, j] = ca
            gi[cp] = ca
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                gi[cp[0] + di, cp[1] + dj] = cb
            succ += 1
            indss -= bd
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Background: the generator paints the whole canvas first, and the crosses are tiny,
    # so the most common colour is reliable here.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Find each cross: a centre cell (colour ca) whose 4 orthogonal neighbours all share
    # one other non-background colour cb.
    crosses = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            ca = int(I[r, c])
            if ca == bgc:
                continue
            arms = {int(I[r - 1, c]), int(I[r + 1, c]), int(I[r, c - 1]), int(I[r, c + 1])}
            if len(arms) == 1:
                cb = arms.pop()
                if cb != bgc and cb != ca:
                    crosses.append((r, c, ca, cb))

    ops, sels = [], []
    for (r, c, ca, cb) in crosses:
        # 1. Grow the cross by 1 pixel north/south/east/west, in the cross colour.
        tips = [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]
        tips = [(i, j) for (i, j) in tips if 0 <= i < h and 0 <= j < w]
        if tips:
            ops.append(cb)
            sels.append(sel_of(tips))
        # 2. Grow the centre pixel 2 pixels along each of the 4 diagonals,
        #    in the centre colour (one ray per op).
        for (dr, dc) in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            ray = [(r + k * dr, c + k * dc) for k in (1, 2)]
            ray = [(i, j) for (i, j) in ray if 0 <= i < h and 0 <= j < w]
            if ray:
                ops.append(ca)
                sels.append(sel_of(ray))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0962bcdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0962bcdd"
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
                                f"for task 0962bcdd"
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
                    f"Failed to build a complete episode for task 0962bcdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0962bcdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
