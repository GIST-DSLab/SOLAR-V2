"""
ARC Task: 9f236235 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    frontcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "frontcol": frontcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, frontcol=None) -> dict:
    from random import choice, sample
    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    if frontcol is None:
        frontcol = choice([c for c in cols if c != bgc])

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi_ = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi_), max(lo, hi_))

    numh = unifint(diff_lb, diff_ub, (2, max(2, min(14, (max_h + 1) // 2))))
    numw = unifint(diff_lb, diff_ub, (2, max(2, min(14, (max_w + 1) // 2))))
    h = unifint(diff_lb, diff_ub, (1, max(1, (max_h + 1) // numh - 1)))
    w = unifint(diff_lb, diff_ub, (1, max(1, (max_w + 1) // numw - 1)))
    remcols = [c for c in cols if c != frontcol]
    numcells = unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = [(a, b) for a in range(numh) for b in range(numw)]
    locs = set(sample(cands, numcells))
    H = h * numh + numh - 1
    W = w * numw + numw - 1
    gi = np.full((H, W), frontcol, dtype=int)
    go = np.full((numh, numw), bgc, dtype=int)
    for (a, b) in cands:
        col = choice(remcols) if (a, b) in locs else bgc
        gi[a * (h + 1):a * (h + 1) + h, b * (w + 1):b * (w + 1) + w] = col
        go[a, b] = col
    go = go[:, ::-1]
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (golf): the frontier lines fix the cell stride; mirror the whole
    # picture left<->right first, then take one pixel per cell block.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # frontier rows/cols are the only fully uniform lines (cell colours never
    # equal the frontier colour, and every block row/col crosses a frontier)
    def uniform_rows(G):
        return [r for r in range(G.shape[0]) if len(set(G[r].tolist())) == 1]
    fr = uniform_rows(I)
    fc = uniform_rows(I.T)
    h = fr[0]
    w = fc[0]
    numh = (hi + 1) // (h + 1)
    numw = (wi + 1) // (w + 1)
    ho, wo = numh, numw

    ops, sels = [], []

    # 1. mirror the whole input left<->right (whole grid = exact rectangle)
    F = I[:, ::-1].copy()
    if not np.array_equal(F, I):
        ops.append(26); sels.append([0, 0, hi - 1, wi - 1])
    else:
        F = I.copy()

    # 2. one pixel per block of the mirrored picture
    S = np.array([[F[a * (h + 1), b * (w + 1)] for b in range(numw)]
                  for a in range(numh)], dtype=int)

    # 3. shrink canvas to the cell-grid size (transparent: zeros stay zero)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = F[:ho, :wo].copy()

    # 4. write each block's sampled colour, one Color op per colour class
    by_color = {}
    for r in range(ho):
        for c in range(wo):
            if cur[r, c] != S[r, c]:
                by_color.setdefault(int(S[r, c]), []).append((r, c))
    bg = Counter(S.flatten().tolist()).most_common(1)[0][0]
    for col in sorted(by_color, key=lambda x: (x != bg, x)):
        ops.append(col); sels.append(sel_of(by_color[col]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 9f236235"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9f236235"
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
                                f"for task 9f236235"
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
                    f"Failed to build a complete episode for task 9f236235 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9f236235-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
