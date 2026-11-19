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
import random
import numpy as np
from maker.sel_helpers import sel_of


def unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + round((b - a) * diff_lb)
    hi = a + round((b - a) * diff_ub)
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc, ringc, inc = random.sample(range(10), 3)
    return {"bgc": bgc, "ringc": ringc, "inc": inc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ringc, inc) -> dict:
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, c0, r1, c1 = loci, locj, loci + ih - 1, locj + iw - 1
    c = np.full((h, w), bgc, dtype=int)
    c[r0:r1 + 1, c0:c1 + 1] = inc
    ring1 = {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)
             if i in (r0, r1) or j in (c0, c1)}
    ring2 = {(i, j) for i in range(r0 + 1, r1) for j in range(c0 + 1, c1)
             if i in (r0 + 1, r1 - 1) or j in (c0 + 1, c1 - 1)}
    for (i, j) in ring1 | ring2:
        c[i, j] = ringc
    corners2 = {(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c0 + 1), (r1 - 1, c1 - 1)}
    cands = sorted(ring2 - corners2)
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = random.sample(cands, numc)
    gi = c.copy()
    for (i, j) in locs:
        gi[i, j] = inc
    lm = r1 - 1
    hori = [ij for ij in locs if r0 + 1 < ij[0] < lm]
    verti = [ij for ij in locs if ij not in hori]
    full = set()
    for (i, j) in hori:
        full |= {(i, jj) for jj in range(w)}
    for (i, j) in verti:
        full |= {(ii, j) for ii in range(h)}
    go = c.copy()
    for (i, j) in full:
        if c[i, j] == bgc:
            go[i, j] = inc
        elif c[i, j] == inc:
            go[i, j] = ringc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    import numpy as np
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # roles by bbox area: background spans the grid, ring colour spans the frame,
    # inner colour lives inside it
    areas = {}
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        areas[int(col)] = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
    order = sorted(areas, key=lambda k: areas[k])
    inc, ringc, bgc = order[0], order[1], order[-1]
    rs, cs = np.where(I == ringc)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    inner_ring = [(i, j) for i in range(r0 + 1, r1) for j in range(c0 + 1, c1)
                  if i in (r0 + 1, r1 - 1) or j in (c0 + 1, c1 - 1)]
    pokes = [(i, j) for (i, j) in inner_ring if I[i, j] == inc]
    rows = sorted({i for (i, j) in pokes if r0 + 1 < i < r1 - 1})   # side pokes -> horizontal beams
    cols = sorted({j for (i, j) in pokes if i in (r0 + 1, r1 - 1)})  # top/bottom pokes -> vertical beams

    ops, sels = [], []
    # 1. region: the whole inner ring is restored to the ring colour (absorbs every poke)
    ops.append(ringc); sels.append(sel_of(inner_ring))

    # 2. colour role "ring": every beam cell crossing the frame's interior, in one stroke
    interior = set()
    for i in rows:
        interior |= {(i, j) for j in range(c0 + 2, c1 - 1)}
    for j in cols:
        interior |= {(i, j) for i in range(r0 + 2, r1 - 1)}
    ops.append(ringc); sels.append(sel_of(sorted(interior)))

    # 3. colour role "inner": every beam cell outside the frame (on background), in one stroke
    outside = set()
    for i in rows:
        outside |= {(i, j) for j in range(w) if j < c0 or j > c1}
    for j in cols:
        outside |= {(i, j) for i in range(h) if i < r0 or i > r1}
    if outside:
        ops.append(inc); sels.append(sel_of(sorted(outside)))

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
