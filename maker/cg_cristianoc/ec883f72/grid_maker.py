"""
ARC Task: ec883f72 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, linc = random.sample(list(range(10)), 3)
    return {"bgc": bgc, "sqc": sqc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or sqc is None or linc is None:
        bgc, sqc, linc = random.sample(list(range(10)), 3)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    ohi = unifint(diff_lb, diff_ub, (0, h - 6))
    owi = unifint(diff_lb, diff_ub, (0, w - 6))
    oh = h - 5 - ohi
    ow = w - 5 - owi
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = sqc
    # frame = outbox(outbox(obj))
    fr0, fr1, fc0, fc1 = loci - 2, loci + oh + 1, locj - 2, locj + ow + 1
    for r in range(fr0, fr1 + 1):
        for c in (fc0, fc1):
            if 0 <= r < h and 0 <= c < w:
                gi[r, c] = linc
    for c in range(fc0, fc1 + 1):
        for r in (fr0, fr1):
            if 0 <= r < h and 0 <= c < w:
                gi[r, c] = linc
    go = gi.copy()
    starts = [((fr1, fc1), (1, 1)), ((fr0, fc0), (-1, -1)),
              ((fr1, fc0), (1, -1)), ((fr0, fc1), (-1, 1))]
    for (r, c), (dr, dc) in starts:
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                # shoot continues off-grid conceptually; positions may re-enter? no: monotone
                if (dr > 0 and r >= h) or (dr < 0 and r < 0) or (dc > 0 and c >= w) or (dc < 0 and c < 0):
                    # may still be negative-start entering grid; handle by continuing while could enter
                    if (dr > 0 and r < 0) or (dc > 0 and c < 0):
                        continue
                    break
                continue
            if gi[r, c] == bgc:
                go[r, c] = sqc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background = colour whose bounding box is largest; frame = next largest
    areas = {}
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        areas[int(col)] = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
    order = sorted(areas, key=lambda c: (-areas[c], c))
    bgc, linc = order[0], order[1]

    def at(r, c):
        return int(I[r, c]) if 0 <= r < h and 0 <= c < w else None

    # Reading: every visible frame corner (a frame cell with exactly one
    # horizontal and one vertical frame neighbour, inward diagonal not frame)
    # launches a diagonal ray outward, coloured like the inner block's corner,
    # which sits two cells diagonally inward from the frame corner.
    ops, sels = [], []
    for r in range(h):
        for c in range(w):
            if I[r, c] != linc:
                continue
            hn = [d for d in (1, -1) if at(r, c + d) == linc]
            vn = [d for d in (1, -1) if at(r + d, c) == linc]
            if len(hn) != 1 or len(vn) != 1:
                continue
            l, J = vn[0], hn[0]           # inward direction
            if at(r + l, c + J) == linc:
                continue
            ray_col = at(r + 2 * l, c + 2 * J)
            if ray_col is None:
                continue
            cells = []
            rr, cc = r - l, c - J
            while 0 <= rr < h and 0 <= cc < w:
                if I[rr, cc] == bgc:
                    cells.append((rr, cc))
                rr -= l
                cc -= J
            if cells:
                ops.append(int(ray_col))
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ec883f72"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ec883f72"
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
                                f"for task ec883f72"
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
                    f"Failed to build a complete episode for task ec883f72 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ec883f72-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
