"""
ARC Task: 77fdfe62 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    c1, c2, c3, c4, barc, bgc, inc = random.sample(list(range(10)), 7)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "barc": barc, "bgc": bgc, "inc": inc}


def generate(diff_lb, diff_ub, max_h, max_w, c1, c2, c3, c4, barc, bgc, inc) -> dict:
    hmax = max(1, min(13, (max_h - 4) // 2))
    wmax = max(1, min(13, (max_w - 4) // 2))
    h = _unifint(diff_lb, diff_ub, (1, hmax))
    w = _unifint(diff_lb, diff_ub, (1, wmax))

    fullh, fullw = 2 * h + 4, 2 * w + 4
    inds = [(i, j) for i in range(h) for j in range(w)]

    ns = [_unifint(diff_lb, diff_ub, (1, h * w)) for _ in range(4)]
    quads = [random.sample(inds, n) for n in ns]

    gi = [[bgc] * fullw for _ in range(fullh)]
    for j in range(fullw):
        gi[1][j] = barc
        gi[fullh - 2][j] = barc
    for i in range(fullh):
        gi[i][1] = barc
        gi[i][fullw - 2] = barc
    gi[0][0] = c1
    gi[0][fullw - 1] = c2
    gi[fullh - 1][0] = c3
    gi[fullh - 1][fullw - 1] = c4

    go = [[bgc] * (2 * w) for _ in range(2 * h)]
    offs = [(0, 0), (0, w), (h, 0), (h, w)]
    qcols = [c1, c2, c3, c4]
    for q in range(4):
        dr, dc = offs[q]
        for (i, j) in quads[q]:
            gi[2 + dr + i][2 + dc + j] = inc
            go[dr + i][dc + j] = qcols[q]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    h, w = ho // 2, wo // 2

    ops, sels = [], []

    # crop away the 2-cell frame -> interior 2h x 2w pattern block
    ops.append(33); sels.append([2, 2, ho - 1, wo - 1])

    interior = I[2:2 + ho, 2:2 + wo]
    bgc = int(I[0, 2])
    incs = [v for v in np.unique(interior).tolist() if v != bgc]
    inc = incs[0] if incs else bgc

    corners = [(int(I[0, 0]), 0, 0), (int(I[0, wi - 1]), 0, w),
               (int(I[hi - 1, 0]), h, 0), (int(I[hi - 1, wi - 1]), h, w)]

    for col, r0, c0 in corners:
        cells = [(r0 + r, c0 + c)
                 for r in range(h) for c in range(w)
                 if interior[r0 + r, c0 + c] == inc]
        if cells:
            ops.append(int(col)); sels.append(sel_of(cells))

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 77fdfe62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 77fdfe62"
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
                                f"for task 77fdfe62"
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
                    f"Failed to build a complete episode for task 77fdfe62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"77fdfe62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
