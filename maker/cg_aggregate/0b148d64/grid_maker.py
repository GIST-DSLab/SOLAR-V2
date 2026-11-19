"""
ARC Task: 0b148d64 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of  # noqa: F401  (selections below are full rectangles)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cola = random.choice(rem)
    colb = random.choice([c for c in rem if c != cola])
    return {"bgc": bgc, "cola": cola, "colb": colb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cola=None, colb=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if bgc is None:
        bgc = random.randint(0, 9)
    if cola is None:
        cola = random.choice([c for c in range(10) if c != bgc])
    if colb is None:
        colb = random.choice([c for c in range(10) if c not in (bgc, cola)])

    H = max(7, min(30, max_h))
    W = max(7, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (7, H))
    w = unifint(diff_lb, diff_ub, (7, W))
    g = [[bgc] * w for _ in range(h)]
    x = random.randint(3, h - 3)
    y = random.randint(3, w - 3)
    di = random.randint(2, h - x - 1)
    dj = random.randint(2, w - y - 1)

    def box(r0, c0, r1, c1):
        return (r0, c0, r1, c1)

    A = box(0, 0, x, y)
    B = box(x + di, 0, h - 1, y)
    C = box(0, y + dj, x, w - 1)
    D = box(x + di, y + dj, h - 1, w - 1)
    boxes = [A, B, C, D]
    trg = random.choice(boxes)
    rem = [b for b in boxes if b != trg]

    def cells(bx):
        r0, c0, r1, c1 = bx
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def subf(bx):
        r0, c0, r1, c1 = bx
        return {
            (r0, random.randint(c0, c1)),
            (r1, random.randint(c0, c1)),
            (random.randint(r0, r1), c0),
            (random.randint(r0, r1), c1),
        }

    def sampler(bx):
        cs = cells(bx)
        k = len(cs) - unifint(diff_lb, diff_ub, (0, len(cs) - 1))
        return set(random.sample(cs, k))

    for (r, c) in sampler(trg) | subf(trg):
        g[r][c] = cola
    for bx in rem:
        for (r, c) in sampler(bx) | subf(bx):
            g[r][c] = colb

    r0, c0, r1, c1 = trg
    go = [row[c0:c1 + 1] for row in g[r0:r1 + 1]]
    return {"input": g, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Reading (golf): the odd colour occupies only one block. Keep the band of
    # ROWS that carry that colour, then the band of COLUMNS that carry it.
    # The odd colour is the one whose cells span the smallest bounding box
    # (background spans the whole grid through the separator bands, and the
    # shared colour spans at least two blocks).
    best = None
    for color in np.unique(I):
        rs, cs = np.where(I == color)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if best is None or area < best[0]:
            best = (area, r0, r1, c0, c1)
    _, r0, r1, c0, c1 = best
    ch, cw = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # 1) Keep only the rows containing the odd colour: full-width rectangle band.
    ops.append(33); sels.append([r0, 0, ch - 1, wi - 1])
    # 2) In the cropped band, keep only the columns containing the odd colour:
    #    full-height rectangle band of the now ch x wi grid.
    ops.append(33); sels.append([0, c0, ch - 1, cw - 1])
    ops.append(34); sels.append([0, 0, ch - 1, cw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0b148d64"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0b148d64"
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
                                f"for task 0b148d64"
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
                    f"Failed to build a complete episode for task 0b148d64 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0b148d64-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
