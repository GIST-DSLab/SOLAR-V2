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


def unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, linc = random.sample(range(10), 3)
    return {"bgc": bgc, "sqc": sqc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, linc=2) -> dict:
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
    # outbox(outbox(obj)): the rectangle outline two cells outside the block
    fr0, fr1, fc0, fc1 = loci - 2, loci + oh + 1, locj - 2, locj + ow + 1
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if (r in (fr0, fr1) or c in (fc0, fc1)) and 0 <= r < h and 0 <= c < w:
                gi[r, c] = linc
    go = gi.copy()
    for (sr, sc), (dr, dc) in [((fr1, fc1), (1, 1)), ((fr0, fc0), (-1, -1)),
                               ((fr1, fc0), (1, -1)), ((fr0, fc1), (-1, 1))]:
        r, c = sr, sc
        while True:
            r += dr; c += dc
            if not (-1 <= r <= h and -1 <= c <= w) and (r < -1 or r > h or c < -1 or c > w):
                if (dr > 0 and r >= h) or (dr < 0 and r < 0) or (dc > 0 and c >= w) or (dc < 0 and c < 0):
                    break
            if (dr > 0 and r >= h) or (dr < 0 and r < 0) or (dc > 0 and c >= w) or (dc < 0 and c < 0):
                break
            if 0 <= r < h and 0 <= c < w and gi[r, c] == bgc:
                go[r, c] = sqc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Colour roles from I: rank colour groups by bounding-box area.
    # Biggest = background (spans the canvas), next = frame, smallest = inner block.
    areas, boxes = {}, {}
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        boxes[int(col)] = (r0, r1, c0, c1)
        areas[int(col)] = (r1 - r0 + 1) * (c1 - c0 + 1)
    order = sorted(areas, key=lambda c: (-areas[c], c))
    bgc, sqc = order[0], order[2]

    # Reading taken from the golf entry: each ray is the diagonal that carries on
    # from a corner of the INNER BLOCK. It goes through the background gap and the
    # frame corner, then out across the background. The inner block is always
    # fully inside the grid, while the frame can be cut off at an edge, so the
    # block's corners are the reliable anchors.
    # The four corners are taken in rotation order (one quarter-turn each, as in
    # the golf loop). Each ray is painted with one Color op in the block's colour.
    br0, br1, bc0, bc1 = boxes[sqc]
    corners = [
        ((br0, bc0), (-1, -1)),  # upper-left
        ((br0, bc1), (-1, 1)),   # upper-right
        ((br1, bc1), (1, 1)),    # lower-right
        ((br1, bc0), (1, -1)),   # lower-left
    ]
    ops, sels = [], []
    for (r, c), (dr, dc) in corners:
        # skip the gap cell and the frame corner (2 diagonal steps beyond the block)
        r += 3 * dr
        c += 3 * dc
        cells = []
        while 0 <= r < h and 0 <= c < w:
            if I[r, c] == bgc:
                cells.append((r, c))
            r += dr
            c += dc
        if cells:
            ops.append(int(sqc))
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
