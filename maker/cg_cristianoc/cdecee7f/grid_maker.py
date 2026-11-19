"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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
    # Only the background matters to the rule; mark colours are read from I.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    h = max(3, min(h, max_h))
    w = max(3, min(w, max_w))
    numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    numc = max(1, min(numc, 9, w))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    numcols = max(1, min(numcols, 9))
    ccols = random.sample(remcols, numcols)
    locs = sorted(random.sample(range(w), numc))
    gi = [[bgc] * w for _ in range(h)]
    go = []
    for j in locs:
        iloc = random.randint(0, h - 1)
        col = random.choice(ccols)
        gi[iloc][j] = col
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = [go[:3], go[3:6][::-1], go[6:]]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # marks are sparse (<= w cells in h*w, h >= 3) -> background is the majority
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # one mark per column, read left to right
    marks = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
    marks.sort(key=lambda rc: rc[1])
    marks = marks[:9]

    # serpentine slot order: row0 L->R, row1 R->L, row2 L->R
    slots = [(0, 0), (0, 1), (0, 2),
             (1, 2), (1, 1), (1, 0),
             (2, 0), (2, 1), (2, 2)]

    # choose a 3x3 patch of canvas to gather the marks into (prefer a mark-free one)
    best = None
    for r0 in range(hi - 2):
        for c0 in range(wi - 2):
            cnt = sum(1 for (r, c) in marks if r0 <= r < r0 + 3 and c0 <= c < c0 + 3)
            if best is None or cnt < best[0]:
                best = (cnt, r0, c0)
            if cnt == 0:
                break
        if best[0] == 0:
            break
    cnt, r0, c0 = best
    region = [r0, c0, 2, 2]   # bbox == exactly the full 3x3 answer patch

    ops, sels = [], []

    if cnt == 0:
        # Gather: slide each mark (left to right) into its serpentine slot.
        # The patch is all background already; the marks' vacated origins lie
        # outside the patch and are discarded by the final crop.
        for k, (r, c) in enumerate(marks):
            col = int(I[r, c])
            tr, tc = r0 + slots[k][0], c0 + slots[k][1]
            if col == 0:
                # ARCLE objects carry only nonzero cells: a 0-mark can't be moved,
                # so lay its colour into the slot directly.
                ops.append(0); sels.append(sel_of([(tr, tc)]))
                continue
            steps = []
            dr, dc = tr - r, tc - c
            steps += [21 if dr > 0 else 20] * abs(dr)
            steps += [22 if dc > 0 else 23] * abs(dc)
            for i, op in enumerate(steps):
                ops.append(op)
                sels.append(sel_of([(r, c)]) if i == 0 else sel_of([]))
        ops.append(33); sels.append(region)
    else:
        # Crowded tiny grid: no free patch, paint the answer patch directly.
        colors = [int(I[r, c]) for r, c in marks]
        colors = colors + [bgc] * (9 - len(colors))
        if any(int(I[r0 + a, c0 + b]) != bgc for a in range(3) for b in range(3)):
            ops.append(bgc); sels.append(region)
        for k in range(9):
            col = colors[k]
            if col == bgc:
                continue
            a, b = slots[k]
            ops.append(col); sels.append(sel_of([(r0 + a, c0 + b)]))
        if not (hi == 3 and wi == 3 and r0 == 0 and c0 == 0):
            ops.append(33); sels.append(region)

    ops.append(34)
    sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
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
                                f"for task cdecee7f"
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
                    f"Failed to build a complete episode for task cdecee7f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
