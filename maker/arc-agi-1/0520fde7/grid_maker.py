"""
ARC Task: 0520fde7 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from maker.sel_helpers import sel_of

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (0, 2)]
    barcol = choice(tuple(cols))
    rem = [c for c in cols if c != barcol]
    cola = choice(tuple(rem))
    colb = choice(tuple(rem))

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(choice(tuple(range(len(VARIANTS)))) and VARIANTS[0] or VARIANTS[0])
                     for _ in range(0)]
        while len(examples) < n_ex:
            examples.append(dict(VARIANTS[len(examples) % len(VARIANTS)]))
        # shuffle deterministically-ish via sample
        examples = [dict(e) for e in sample(tuple(examples), len(examples))]
    else:
        examples = [dict(e) for e in sample(tuple(VARIANTS), n_ex)]
    plan = examples + [dict(examples[choice(tuple(range(len(examples))))])]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             barcol=None, cola=None, colb=None, mirrored=None) -> dict:
    bgc = 0
    cols = [c for c in range(10) if c not in (0, 2)]
    if barcol is None:
        barcol = choice(tuple(cols))
    if cola is None:
        cola = choice(tuple([c for c in cols if c != barcol]))
    if colb is None:
        colb = choice(tuple([c for c in cols if c != barcol]))
    if mirrored is None:
        mirrored = choice((True, False))

    # gi is (h, 2w+1); after dmirror it is (2w+1, h)
    if mirrored:
        h_ub = min(30, max_w)
        w_ub = min(14, (max_h - 1) // 2)
    else:
        h_ub = min(30, max_h)
        w_ub = min(14, (max_w - 1) // 2)
    h_ub = max(2, h_ub)
    w_ub = max(2, w_ub)

    h = unifint(diff_lb, diff_ub, (2, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))

    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 2, set(a) & set(b))
    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (measured from I only):
    Locate the one-colour separator bar splitting I into two equal halves.
    A cell of the first half becomes 2 when the mirrored cell of the second
    half is also marked, otherwise it is cleared. Then crop to that half.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- locate the separator: the middle column if it is uniform, else the middle row
    whalf = wi // 2
    hhalf = hi // 2
    vertical = len(set(I[:, whalf].tolist())) == 1

    if vertical:
        cells = [(r, c) for r in range(hi) for c in range(whalf)]
        def partner(r, c):
            return (r, wi - whalf + c)
        crop_h, crop_w = hi, whalf
        crop_sel = [0, 0, hi - 1, whalf - 1]      # full rectangle = the first half
    else:
        cells = [(r, c) for r in range(hhalf) for c in range(wi)]
        def partner(r, c):
            return (hi - hhalf + r, c)
        crop_h, crop_w = hhalf, wi
        crop_sel = [0, 0, hhalf - 1, wi - 1]      # full rectangle = the first half

    both, only = [], []
    for (r, c) in cells:
        pr, pc = partner(r, c)
        if I[r, c] != 0:
            if I[pr, pc] != 0:
                both.append((r, c))
            else:
                only.append((r, c))

    # 1. mark coincidences with 2
    if both:
        ops.append(2)
        sels.append(sel_of(both))
    # 2. clear marks of the first half that have no counterpart
    if only:
        ops.append(0)
        sels.append(sel_of(only))
    # 3. keep only the first half (bbox = exactly that rectangular region)
    ops.append(33)
    sels.append(crop_sel)

    ops.append(34)
    sels.append([0, 0, crop_h - 1, crop_w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0520fde7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0520fde7"
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
                                f"for task 0520fde7"
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
                    f"Failed to build a complete episode for task 0520fde7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0520fde7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
