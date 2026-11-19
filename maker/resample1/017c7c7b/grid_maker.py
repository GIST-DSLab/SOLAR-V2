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
from maker.sel_helpers import sel_of


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return int(round(a + d * (b - a)))


def _period(I):
    # smallest vertical period of the grid (rows r and r+p identical everywhere)
    h = I.shape[0]
    for p in range(1, h + 1):
        if all((I[r] == I[r + p]).all() for r in range(h - p)):
            return p
    return h


def _shift(h, p):
    # smallest multiple of the period that reaches back at least h//2 rows
    s = p
    while s < h // 2:
        s += p
    return s


def sample_colors(num_examples=None) -> dict:
    fgc = random.choice([c for c in range(10) if c not in (0, 2)])
    return {"fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, fgc=None, **kw) -> dict:
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c not in (0, 2)])
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (3, max(3, min(10, max_h // 3))))
        w = _unifint(diff_lb, diff_ub, (2, max(2, min(30, max_w))))
        h += h
        H = h + h // 2
        go = np.zeros((H, w), dtype=int)
        oh = _unifint(diff_lb, diff_ub, (2, max(2, h // 3 * 2)))
        ow = _unifint(diff_lb, diff_ub, (1, w))
        locj = random.randint(0, w - ow)
        bounds = [(r, c) for r in range(oh) for c in range(ow)]
        ncells = min(max(1, _unifint(diff_lb, diff_ub, (1, oh * ow))), oh * ow)
        obj = random.sample(bounds, ncells)
        for k in range((2 * h) // oh):
            for r, c in obj:
                rr = r + k * oh
                if rr < H:
                    go[rr, c + locj] = 2
        gi = go[:h].copy()
        gi[gi == 2] = fgc
        # reject instances whose continuation is ambiguous from the input alone
        p = _period(gi)
        s = _shift(h, p)
        pred = (gi != 0).astype(int) * 2
        pred = np.vstack([pred, pred[h - s:h - s + h // 2]])
        if pred.shape == go.shape and (pred == go).all():
            return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("failed to generate unambiguous instance")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    # 1. recolour every foreground cell to 2 (output colour is fixed by the task)
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != 0]
    if fg:
        ops.append(2); sels.append(sel_of(fg))
    # 2. extend canvas downward by half the height (bg = 0, so padding is already bg)
    H = h + h // 2
    ops.append(33); sels.append([0, 0, H - 1, w - 1])
    # 3. continue the vertical period: copy a period-aligned band of full rows
    #    (whole-width rectangle intended) from the working grid and paste it below
    p = _period(I)
    s = _shift(h, p)
    src_r = h - s
    band = I[src_r:src_r + h // 2]
    if (band != 0).any():
        ops.append(29); sels.append([src_r, 0, h // 2 - 1, w - 1])
        ops.append(30); sels.append([h, 0, 0, 0])
    ops.append(34); sels.append([0, 0, H - 1, w - 1])
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
