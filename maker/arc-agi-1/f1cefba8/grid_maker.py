"""
ARC Task: f1cefba8 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ringc, inc = random.sample(cols, 3)
    return {"bgc": bgc, "ringc": ringc, "inc": inc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, ringc: int, inc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (7, max_h))
    w = unifint(diff_lb, diff_ub, (7, max_w))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    obj = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    ring1 = box(obj)
    ring2 = inbox(obj)
    bd = backdrop(obj)
    c = canvas(bgc, (h, w))
    c = fill(c, inc, bd)
    c = fill(c, ringc, ring1 | ring2)
    cands = totuple(ring2 - corners(ring2))
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = sample(cands, numc)
    gi = fill(c, inc, locs)
    lm = lowermost(ring2)
    hori = sfilter(locs, lambda ij: ij[0] > loci + 1 and ij[0] < lm)
    verti = difference(locs, hori)
    hlines = mapply(hfrontier, hori)
    vlines = mapply(vfrontier, verti)
    fulllocs = set(hlines) | set(vlines)
    topaintinc = fulllocs & ofcolor(c, bgc)
    topaintringc = fulllocs & ofcolor(c, inc)
    go = fill(c, inc, topaintinc)
    go = fill(go, ringc, topaintringc)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # --- roles: bgc spans whole grid, ringc spans the frame box, inc lives inside it
    areas = {}
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        areas[int(col)] = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
    order = sorted(areas, key=lambda k: areas[k])
    inc, ringc, bgc = order[0], order[1], order[-1]

    rs, cs = np.where(I == ringc)
    r0, r1 = int(rs.min()), int(rs.max())   # outer frame rows
    c0, c1 = int(cs.min()), int(cs.max())   # outer frame cols

    # --- pokes: inc-colored cells sitting on the inner ring (rows r0+1/r1-1, cols c0+1/c1-1)
    pokes = set()
    for j in range(c0 + 1, c1):
        for i in (r0 + 1, r1 - 1):
            if I[i, j] == inc:
                pokes.add((i, j))
    for i in range(r0 + 2, r1 - 1):
        for j in (c0 + 1, c1 - 1):
            if I[i, j] == inc:
                pokes.add((i, j))

    row_lines = sorted({i for (i, j) in pokes if r0 + 1 < i < r1 - 1})   # side pokes -> beam across
    col_lines = sorted({j for (i, j) in pokes if i in (r0 + 1, r1 - 1)})  # top/bottom pokes -> beam down

    ops, sels = [], []

    # each side poke shoots a horizontal beam: inside the box the beam is ring-colored
    # (swallowing the poke itself), outside the box it is inc-colored
    for i in row_lines:
        lo = c0 + 1 if (i, c0 + 1) in pokes else c0 + 2
        hi = c1 - 1 if (i, c1 - 1) in pokes else c1 - 2
        ops.append(ringc); sels.append([i, lo, 0, hi - lo])
        if c0 > 0:
            ops.append(inc); sels.append([i, 0, 0, c0 - 1])
        if c1 < w - 1:
            ops.append(inc); sels.append([i, c1 + 1, 0, w - 2 - c1])

    # each top/bottom poke shoots a vertical beam, same colouring rule
    for j in col_lines:
        lo = r0 + 1 if (r0 + 1, j) in pokes else r0 + 2
        hi = r1 - 1 if (r1 - 1, j) in pokes else r1 - 2
        ops.append(ringc); sels.append([lo, j, hi - lo, 0])
        if r0 > 0:
            ops.append(inc); sels.append([0, j, r0 - 1, 0])
        if r1 < h - 1:
            ops.append(inc); sels.append([r1 + 1, j, h - 2 - r1, 0])

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f1cefba8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f1cefba8"
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
                                f"for task f1cefba8"
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
                    f"Failed to build a complete episode for task f1cefba8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f1cefba8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
