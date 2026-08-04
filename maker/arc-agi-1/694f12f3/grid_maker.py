"""
ARC Task: 694f12f3 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        cells = list(cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        return random.randint(int(round(a + (b - a) * lb)),
                              int(round(a + (b - a) * ub)))

    lim = min(max_h, max_w)          # rotation may swap h/w
    if lim < 9:
        lim = 9
    h = unifint(diff_lb, diff_ub, (9, lim))
    w = unifint(diff_lb, diff_ub, (9, lim))

    seploc = random.randint(4, h - 5)
    bigh = unifint(diff_lb, diff_ub, (4, seploc))
    bigw = unifint(diff_lb, diff_ub, (3, w - 1))
    bigloci = random.randint(0, seploc - bigh)
    biglocj = random.randint(0, w - bigw)

    smallmaxh = h - seploc - 1
    smallmaxw = w - 1
    bigsize = bigh * bigw
    cands = []
    for a in range(3, smallmaxh + 1):
        for b in range(3, smallmaxw + 1):
            if a * b < bigsize:
                cands.append((a, b))
    if not cands:
        raise ValueError("no small rect candidates")
    cands.sort(key=lambda ab: ab[0] * ab[1])
    idx = unifint(diff_lb, diff_ub, (0, len(cands) - 1))
    smallh, smallw = cands[idx]
    smallloci = random.randint(seploc + 1, h - smallh)
    smalllocj = random.randint(0, w - smallw)

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    for r in range(bigloci, bigloci + bigh):
        for c in range(biglocj, biglocj + bigw):
            gi[r][c] = sqc
    for r in range(smallloci, smallloci + smallh):
        for c in range(smalllocj, smalllocj + smallw):
            gi[r][c] = sqc

    go = [row[:] for row in gi]
    for r in range(bigloci + 1, bigloci + bigh - 1):
        for c in range(biglocj + 1, biglocj + bigw - 1):
            go[r][c] = 2
    for r in range(smallloci + 1, smallloci + smallh - 1):
        for c in range(smalllocj + 1, smalllocj + smallw - 1):
            go[r][c] = 1

    k = random.choice((0, 1, 2, 3))
    if k:
        gi = np.rot90(np.array(gi, dtype=int), k).tolist()
        go = np.rot90(np.array(go, dtype=int), k).tolist()

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Only change: each rectangle's interior gets painted (small->1, big->2).
    # Interiors are solid rectangles, measured directly from the I/O diff.
    for val in (1, 2):
        cells = [(r, c) for r in range(ho) for c in range(wo)
                 if O[r, c] == val and I[r, c] != val]
        if not cells:
            continue
        ops.append(val)
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 694f12f3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 694f12f3"
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
                                f"for task 694f12f3"
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
                    f"Failed to build a complete episode for task 694f12f3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"694f12f3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
