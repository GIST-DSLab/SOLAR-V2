"""
ARC Task: 017c7c7b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # generator samples one foreground color from 0..9 minus {0, 2}; background is hardcoded 0
    fgc = random.choice([1, 3, 4, 5, 6, 7, 8, 9])
    return {"fgc": fgc}


def _vperiod_rows(g):
    n = len(g)
    for p in range(1, n):
        if all(g[r] == g[r + p] for r in range(n - p)):
            return p
    return n


def generate(diff_lb, diff_ub, max_h, max_w, fgc=None, **kwargs) -> dict:
    if fgc is None:
        fgc = random.choice([1, 3, 4, 5, 6, 7, 8, 9])
    hcap = max(3, min(10, max_h // 3))          # output height is 3 * (h/2) -> fits max_h
    wcap = max(2, min(30, max_w))
    gi, go = None, None
    for _ in range(64):
        h = unifint(diff_lb, diff_ub, (3, hcap))
        w = unifint(diff_lb, diff_ub, (2, wcap))
        h += h
        go = canvas(0, (h + h // 2, w))
        oh = unifint(diff_lb, diff_ub, (2, max(2, h // 3 * 2)))
        ow = unifint(diff_lb, diff_ub, (1, w))
        locj = randint(0, w - ow)
        bounds = asindices(canvas(-1, (oh, ow)))
        ncellsd = unifint(diff_lb, diff_ub, (1, oh * ow))
        ncells = min(max(1, ncellsd), oh * ow)
        obj = sample(totuple(bounds), ncells)
        obj = {(r, c + locj) for (r, c) in obj}
        for k in range((2 * h) // oh):
            go = fill(go, 2, shift(obj, (k * oh, 0)))
        gi = replace(go[:h], 2, fgc)
        ho = len(go)
        # keep only instances where the period visible in the input really governs the
        # continuation, and where the continued rows actually show something
        p = _vperiod_rows(gi)
        if any(v != 0 for row in go[h:] for v in row) and \
           all(go[r] == go[r - p] for r in range(h, ho)):
            return {'input': gi, 'output': go}
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # vertical period of the pattern, measured on the input alone
    p = hi
    for q in range(1, hi):
        if np.array_equal(I[q:], I[:hi - q]):
            p = q
            break

    # 1) recolour the whole pattern to 2 (its exact cells; background is 0)
    cells = [(int(r), int(c)) for r in range(hi) for c in range(wi) if I[r, c] != 0]
    if cells:
        ops.append(2)
        sels.append(sel_of(cells))

    # 2) grow the canvas downwards to the output height
    #    (bbox selection = exactly the whole new canvas rectangle)
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    # 3) the added rows continue the pattern: they are a copy of the full-width band
    #    lying m rows above them, m = whole number of periods covering the gap.
    #    The gap is all background (0) after the resize, so Paste carries the band
    #    faithfully -- nothing has to be painted cell by cell.
    gap = ho - hi
    if gap > 0:
        m = ((gap + p - 1) // p) * p          # multiple of p, m >= gap, m <= hi
        src_r = hi - m
        if src_r >= 0 and I[src_r:src_r + gap].any():
            ops.append(29)                                   # CopyO: the band, full width
            sels.append([src_r, 0, gap - 1, wo - 1])
            ops.append(30)                                   # Paste at the gap's top-left
            sels.append([hi, 0, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 017c7c7b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 017c7c7b"
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
                                f"for task 017c7c7b"
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
                    f"Failed to build a complete episode for task 017c7c7b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"017c7c7b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
