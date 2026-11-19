"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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
    # Colour 8 is fixed by the rule (it is the alternating stripe colour), so it is never sampled.
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    # In the base frame the grid is h x w. After a 90/270 rotation it becomes w x h.
    if rot % 2 == 1:
        lim_h, lim_w = max_w, max_h
    else:
        lim_h, lim_w = max_h, max_w
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, lim_h))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, lim_w))))
    h = max(3, min(h, lim_h))
    w = max(3, min(w, lim_w))

    barth = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    barth = max(1, min(barth, max(1, w // 5)))
    loci = unifint(diff_lb, diff_ub, (1, h - 2))
    loci = max(1, min(loci, h - 2))
    locj = random.randint(1, w - barth - 1)

    gi = np.full((h, w), bgc, dtype=int)
    gi[0:loci + 1, locj:locj + barth] = fgc

    go = np.full((h, w), bgc, dtype=int)
    for k in range(32):
        color = fgc if k % 2 == 0 else 8
        for s in (1, -1):
            dr = -k
            dc = s * k * barth
            r0, r1 = 0 + dr, loci + dr
            c0, c1 = locj + dc, locj + barth - 1 + dc
            a, b = max(0, r0), min(h - 1, r1)
            c, d = max(0, c0), min(w - 1, c1)
            if a <= b and c <= d:
                go[a:b + 1, c:d + 1] = color

    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    mask = I != bgc
    ops, sels = [], []
    if not mask.any():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    rs, cs = np.where(mask)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    fgc = int(I[r0, c0])

    # The bar is attached to exactly one grid edge.
    if r0 == 0:
        edge = 'top'
    elif r1 == hi - 1:
        edge = 'bottom'
    elif c0 == 0:
        edge = 'left'
    else:
        edge = 'right'

    if edge in ('top', 'bottom'):
        barth = c1 - c0 + 1
        dr_unit, dc_unit = (-1 if edge == 'top' else 1), 0
        perp_r, perp_c = 0, barth
    else:
        barth = r1 - r0 + 1
        dr_unit, dc_unit = 0, (-1 if edge == 'left' else 1)
        perp_r, perp_c = barth, 0

    # Grow the staircase on one side of the bar first, then on the other side.
    # Each new stripe is the bar shifted sideways by k*barth and pulled back
    # toward its edge by k, so it is one cell shorter than the stripe before it.
    # Stripe colours alternate 8 / fgc. Each stripe is a full rectangle.
    for side in (1, -1):
        k = 1
        while True:
            dr = k * dr_unit + side * k * perp_r
            dc = k * dc_unit + side * k * perp_c
            a = max(0, r0 + dr); b = min(hi - 1, r1 + dr)
            c = max(0, c0 + dc); d = min(wi - 1, c1 + dc)
            if a > b or c > d:
                break
            color = 8 if k % 2 == 1 else fgc
            ops.append(color); sels.append([a, c, b - a, d - c])  # full stripe rectangle
            k += 1

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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
