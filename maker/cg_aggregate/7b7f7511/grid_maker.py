"""
ARC Task: 7b7f7511 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


VARIANTS = [{"vertical": True}, {"vertical": False}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def _ambiguous(gi):
    gi = np.asarray(gi)
    H, W = gi.shape
    lw, th = W // 2, H // 2
    lr = lw >= 1 and np.array_equal(gi[:, :lw], gi[:, W - lw:])
    tb = th >= 1 and np.array_equal(gi[:th, :], gi[H - th:, :])
    return lr and tb


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, vertical=None, **kwargs) -> dict:
    if vertical is None:
        vertical = random.choice([True, False])
    cols = list(range(10))
    for _ in range(200):
        if vertical:
            # go is h x w, dmirror -> w x h, vconcat -> 2w x h
            h = _unifint(diff_lb, diff_ub, (2, min(30, max_w)))
            w = _unifint(diff_lb, diff_ub, (1, min(15, max_h // 2)))
        else:
            # hconcat -> h x 2w
            h = _unifint(diff_lb, diff_ub, (2, min(30, max_h)))
            w = _unifint(diff_lb, diff_ub, (1, min(15, max_w // 2)))
        w = max(w, 1)
        go = [[bgc] * w for _ in range(h)]
        remcols = [c for c in cols if c != bgc]
        numc = _unifint(diff_lb, diff_ub, (1, max(1, min(9, h * w - 1))))
        colsch = random.sample(remcols, numc)
        inds = [(i, j) for i in range(h) for j in range(w)]
        for col in colsch:
            if not inds:
                break
            num = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
            num = min(num, len(inds))
            chos = random.sample(inds, num)
            for (i, j) in chos:
                go[i][j] = col
            chs = set(chos)
            inds = [x for x in inds if x not in chs]
        if vertical:
            go = [list(r) for r in zip(*go)]  # dmirror
            gi = [list(r) for r in go] + [list(r) for r in go]
        else:
            gi = [list(r) + list(r) for r in go]
        # reject inputs that are doubled both ways: there the two halving orders
        # disagree and the tile is not recoverable from the grid
        if _ambiguous(gi):
            continue
        if len(gi) > max_h or len(gi[0]) > max_w:
            continue
        return {"input": gi, "output": [list(r) for r in go]}
    raise ValueError("could not generate unambiguous instance")


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of  # noqa: F401  (selection here is a full rectangle)
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Reading: the input is ONE tile written twice. Undo the doubling.
    # First ask whether the grid's rows are a list stacked on itself
    # (top half == bottom half); if so the tile is the top half.
    # Otherwise every row is a row doubled side by side, and halving each row
    # is the same as keeping the left half of the grid.
    th = hi // 2
    if th >= 1 and hi % 2 == 0 and np.array_equal(I[:th, :], I[th:, :]):
        ho, wo = th, wi            # top half
    else:
        ho, wo = hi, wi // 2       # left half (each row halved)

    # One CropGrid onto the tile rectangle (top-left, full rectangle intended),
    # whose size is read from I alone.
    ops = [33, 34]
    sels = [[0, 0, ho - 1, wo - 1], [0, 0, ho - 1, wo - 1]]
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
                        f"num_examples+1 ({num_examples + 1}) for task 7b7f7511"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7b7f7511"
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
                                f"for task 7b7f7511"
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
                    f"Failed to build a complete episode for task 7b7f7511 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7b7f7511-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
