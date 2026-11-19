"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    barc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "barc": barc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    def rot90(g):
        return [list(r) for r in zip(*g[::-1])]

    def hconcat(a, b):
        return [list(ra) + list(rb) for ra, rb in zip(a, b)]

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if barc is None or barc == bgc:
        barc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, barc)]

    for _ in range(200):
        d = unifint(diff_lb, diff_ub, (2, 9))
        k = 4 if d < 7 else 3
        num = unifint(diff_lb, diff_ub, (2, k))
        vertical = random.choice((True, False))
        long_side = num * (d + 1) - 1
        if vertical:
            if long_side <= max_h and d <= max_w:
                break
        else:
            if long_side <= max_w and d <= max_h:
                break
    else:
        d, num, vertical = 2, 2, False

    ncols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
    patcols = random.sample(remcols, ncols)
    c = [[bgc] * d for _ in range(d)]
    inds = [(i, j) for i in range(d) for j in range(d)]
    ncells = unifint(diff_lb, diff_ub, (1, d * d - 1))
    for (i, j) in random.sample(inds, ncells):
        c[i][j] = random.choice(patcols)
    bgcanv = [[bgc] * d for _ in range(d)]
    barr = [[barc] for _ in range(d)]
    fillinidx = random.randrange(num)
    gi = [list(r) for r in (c if fillinidx == 0 else bgcanv)]
    go = [list(r) for r in c]
    for j in range(num - 1):
        c = rot90(c)
        gi = hconcat(hconcat(gi, barr), c if j + 1 == fillinidx else bgcanv)
        go = hconcat(hconcat(go, barr), c)
    if vertical:
        gi = rot90(gi)
        go = rot90(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # d x d panels separated by 1-cell bars, laid out along the long axis
    vertical = hi > wi
    if vertical:
        d = wi
        num = (hi + 1) // (d + 1)
        origins = [(j * (d + 1), 0) for j in range(num)]
    else:
        d = hi
        num = (wi + 1) // (d + 1)
        origins = [(0, j * (d + 1)) for j in range(num)]

    panels = [I[r:r + d, c:c + d] for (r, c) in origins]
    src, bgc = 0, None
    for j, p in enumerate(panels):
        if len(set(p.flatten().tolist())) > 1:
            src = j
        else:
            bgc = int(p[0, 0])
    if bgc is None:
        bgc = int(panels[(src + 1) % num][0, 0])

    sr, sc = origins[src]
    has_zero = bool((panels[src] == 0).any())

    ops, sels = [], []
    # The source panel is the single unit; every other panel is that source
    # turned by its distance from it (quarter-turns), read directly off the source.
    ops.append(28); sels.append([sr, sc, d - 1, d - 1])  # full d x d panel rectangle

    for t in range(num):
        if t == src:
            continue
        tr, tc = origins[t]
        box = [tr, tc, d - 1, d - 1]  # whole d x d panel (square)
        if has_zero and bgc != 0:
            ops.append(0); sels.append(box)  # pattern owns 0-cells: blank the panel
        ops.append(30); sels.append([tr, tc, 0, 0])
        kk = (t - src) % 4
        if kk == 1:
            ops.append(25); sels.append(box)       # quarter turn CW
        elif kk == 3:
            ops.append(24); sels.append(box)       # quarter turn CCW
        elif kk == 2:
            # half turn read as reversing rows and reversing order of rows
            ops.append(26); sels.append(box)       # left<->right
            ops.append(27); sels.append(box)       # up<->down

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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
