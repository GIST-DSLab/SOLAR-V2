"""
ARC Task: 23b5c85d (RE-ARC) — LLM-generated grid_maker
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


# Comparing the two readings of the task:
#   maker:  among the colour regions that exactly fill their bounding box, take
#           the one with the smallest area and crop to it.
#   golf:   the answer is a single colour; collect every cell of that colour
#           and lay them out as the output.
# Both treat the answer as "all cells of one colour", and both lead to the same
# single operation: crop the canvas to that colour. The golf picks the colour
# by lowest cell count. That shortcut is not guaranteed by the generator: a
# larger rectangle that is partly covered can show fewer cells than the target.
# So the colour is chosen with the generator's rule (smallest solid rectangle,
# read from I only). The golf reading does change how the selection is named:
# it is now "all cells of colour v" (a colour mask), not a bounding box.
# Because the region is solid, that mask is exactly a full rectangle, so the
# crop result is the same.


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    rng = b - a
    lo = int(round(a + diff_lb * rng))
    hi = int(round(a + diff_ub * rng))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _solid_regions(g):
    """Return {colour: (area, r0, c0, h, w)} for every colour whose cells fill their bbox."""
    g = np.asarray(g)
    out = {}
    for c in set(g.flatten().tolist()):
        rs, cs = np.where(g == c)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if len(rs) == bh * bw:
            out[int(c)] = (int(bh * bw), int(r0), int(c0), int(bh), int(bw))
    return out


def sample_colors(num_examples=None) -> dict:
    # Only the background is fixed per episode; the rectangle colours never
    # decide the rule (the rule depends only on shape/area).
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    max_h = max(5, min(30, max_h))
    max_w = max(5, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (5, max_h))
    w = _unifint(diff_lb, diff_ub, (5, max_w))
    colopts = [c for c in range(10) if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    oh = _unifint(diff_lb, diff_ub, (2, h - 1))
    ow = _unifint(diff_lb, diff_ub, (2, w - 1))
    num = _unifint(diff_lb, diff_ub, (1, 8))
    go = None
    cnt = 0
    while cnt < num and colopts:
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        col = random.choice(colopts)
        colopts = [c for c in colopts if c != col]
        gi2 = gi.copy()
        gi2[loci:loci + oh, locj:locj + ow] = col
        solids = _solid_regions(gi2)
        if col not in solids:
            break
        my_area = solids[col][0]
        # new rectangle must be the strict unique minimum solid region
        if any(a <= my_area for c, (a, *_r) in solids.items() if c != col):
            break
        gi = gi2
        go = np.full((oh, ow), col, dtype=int)
        oh = _unifint(diff_lb, diff_ub, (max(0, oh - 4), oh - 1))
        ow = _unifint(diff_lb, diff_ub, (max(0, ow - 4), ow - 1))
        if oh < 1 or ow < 1:
            break
        cnt += 1
    if go is None:
        raise ValueError("generation failed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)

    # Rule read from I only: each colour taken as one unit (all of its cells);
    # keep the colours whose cells form a solid rectangle; the answer colour is
    # the one with the smallest area.
    solids = _solid_regions(I)
    v = min(solids, key=lambda c: solids[c][0])
    _, r0, c0, ho, wo = solids[v]

    cells = [(int(r), int(c)) for r, c in zip(*np.where(I == v))]

    ops, sels = [], []
    # CropGrid on the mask "all cells of colour v". The canvas becomes that
    # colour's bounding box, which is exactly those cells because the region is solid.
    ops.append(33); sels.append(sel_of(cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task 23b5c85d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23b5c85d"
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
                                f"for task 23b5c85d"
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
                    f"Failed to build a complete episode for task 23b5c85d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23b5c85d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
