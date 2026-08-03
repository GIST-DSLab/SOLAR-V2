"""
ARC Task: a699fb00 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int(round(diff_lb * (b - a)))
    hi = a + int(round(diff_ub * (b - a)))
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # generator: cols = all colors except 2 (2 is the hardcoded fill color)
    cols = [c for c in range(10) if c != 2]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = _unifint(diff_lb, diff_ub, (4, max_h))
    w = _unifint(diff_lb, diff_ub, (4, max_w))
    numls = _unifint(diff_lb, diff_ub, (1, h - 1))
    locs = random.sample(list(range(h)), numls)

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]

    for ii in locs:
        endidx = _unifint(diff_lb, diff_ub, (2, w - 2))
        ofs = _unifint(diff_lb, diff_ub, (1, max(1, endidx // 2))) * 2
        ofs = min(max(2, ofs), endidx)
        startidx = endidx - ofs
        for jj in range(startidx, endidx + 1):
            go[ii][jj] = 2
        for jj in range(startidx, endidx + 1, 2):
            go[ii][jj] = fgc
            gi[ii][jj] = fgc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Rule measured from I alone: in every row that carries fgc dots, the dots sit
    two apart; each one-cell gap between two consecutive dots of the same run is
    painted 2.  O is only used for its shape (identical to I's)."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = the colour the canvas was painted with (dominant); dots are the other colour
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]

    ops, sels = [], []

    for r in range(hi):
        dots = [c for c in range(wi) if I[r, c] != bgc]
        if len(dots) < 2:
            continue
        # split the row's dots into runs whose consecutive spacing is exactly 2
        runs, cur = [], [dots[0]]
        for prev, cc in zip(dots, dots[1:]):
            if cc - prev == 2:
                cur.append(cc)
            else:
                runs.append(cur)
                cur = [cc]
        runs.append(cur)

        for run in runs:
            if len(run) < 2:
                continue
            gaps = [(r, run[k] + 1) for k in range(len(run) - 1)]
            ops.append(2)               # Color2 over this dotted line's internal gaps
            sels.append(sel_of(gaps))

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
                        f"num_examples+1 ({num_examples + 1}) for task a699fb00"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a699fb00"
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
                                f"for task a699fb00"
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
                    f"Failed to build a complete episode for task a699fb00 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a699fb00-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
