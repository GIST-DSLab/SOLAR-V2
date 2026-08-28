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
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # generator samples: bgc, and a palette ccols (numc colors) used for the plus shapes
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(2, 7)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        lo = max(a, min(lo, b))
        hi = max(a, min(hi, b))
        if lo > hi:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = [c for c in range(10) if c not in (3, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(2, 7))
    ccols = list(ccols)

    hub = max(10, min(30, int(max_h)))
    wub = max(10, min(30, int(max_w)))
    hlb = min(10, hub)
    wlb = min(10, wub)

    while True:
        h = unifint(diff_lb, diff_ub, (hlb, hub))
        w = unifint(diff_lb, diff_ub, (wlb, wub))

        gi = [[bgc for _ in range(w)] for _ in range(h)]
        go = [[bgc for _ in range(w)] for _ in range(h)]

        num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
        indss = set((i, j) for i in range(h) for j in range(w))
        subs = [(i, j) for i in range(h) for j in range(w) if i < h - 5 and j < w - 5]

        maxtrials = 4 * num
        tr = 0
        succ = 0
        while succ < num and tr <= maxtrials:
            if len(indss) == 0 or len(subs) == 0:
                break
            loci, locj = random.choice(subs)
            bd = set(
                (loci + di, locj + dj) for di in range(5) for dj in range(5)
            )
            if bd.issubset(indss):
                ca, cb = random.sample(ccols, 2)
                cp = (loci + 2, locj + 2)
                # diagonals of the 5x5 block
                lins12 = set()
                for k in range(5):
                    lins12.add((loci + k, locj + k))
                    lins12.add((loci + 4 - k, locj + k))
                # cross (mid row + mid col) of the 5x5 block
                lins34 = set()
                for k in range(5):
                    lins34.add((loci + 2, locj + k))
                    lins34.add((loci + k, locj + 2))
                for (r, c) in lins34:
                    go[r][c] = cb
                for (r, c) in lins12:
                    go[r][c] = ca
                gi[cp[0]][cp[1]] = ca
                for (dr, dc) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    gi[cp[0] + dr][cp[1] + dc] = cb
                succ += 1
                indss = indss - bd
            tr += 1

        if succ >= 1:
            break

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background: the canvas colour the generator paints before placing plus shapes
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # locate every plus: a centre cell (colour ca) whose 4 orthogonal neighbours share
    # one other non-background colour cb
    plusses = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            ca = int(I[r, c])
            if ca == bgc:
                continue
            arms = {int(I[r - 1, c]), int(I[r + 1, c]), int(I[r, c - 1]), int(I[r, c + 1])}
            if len(arms) == 1:
                cb = arms.pop()
                if cb != bgc and cb != ca:
                    plusses.append((r, c, ca, cb))

    ops, sels = [], []

    for (r, c, ca, cb) in plusses:
        # 1. extend the plus's four arms outward by one cell -> full 5-long cross in cb
        tips = [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]
        ops.append(cb)
        sels.append(sel_of(tips))

        # 2. draw the main diagonal of the 5x5 block in ca (the centre is already ca)
        main_diag = [(r - 2, c - 2), (r - 1, c - 1), (r + 1, c + 1), (r + 2, c + 2)]
        ops.append(ca)
        sels.append(sel_of(main_diag))

        # 3. mirror the whole 5x5 block left<->right: the diagonal becomes the
        #    anti-diagonal (the cross is symmetric, so it is unchanged).
        #    Selection is exactly the full 5x5 rectangle, background included.
        ops.append(26)
        sels.append([r - 2, c - 2, 4, 4])

        # 4. the mirror moved the diagonal away; draw it again so both arms of the X exist
        ops.append(ca)
        sels.append(sel_of(main_diag))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
