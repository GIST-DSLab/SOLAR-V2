"""
ARC Task: a740d043 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background is the only colour role the rule depends on (it becomes 0 in O).
    # Object colours are irrelevant to the rule, so they stay sampled per instance.
    cols = [c for c in range(1, 10)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    from random import randint, choice, sample

    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        lo_i = int(round(lo))
        hi_i = int(round(hi))
        if hi_i < lo_i:
            hi_i = lo_i
        return max(a, min(b, randint(lo_i, hi_i)))

    cols = [c for c in range(1, 10)]
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, ((h - 1) * (w - 1)) // 2)))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, len(remcols)))
    remcols = sample(remcols, numc)

    bounds = {(i, j) for i in range(h - 1) for j in range(w - 1)}
    ch = choice(sorted(bounds))
    shp = {ch}
    bounds.discard(ch)

    def neighbors(p):
        i, j = p
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if not (di == 0 and dj == 0)}

    for _ in range(nc):
        nb = set()
        for p in shp:
            nb |= neighbors(p)
        cand = sorted((bounds - shp) & nb)
        if not cand:
            break
        shp.add(choice(cand))
    mi = min(i for i, _ in shp)
    mj = min(j for _, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, _ in shp) + 1
    ow = max(j for _, j in shp) + 1
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)

    gi = [[bgc] * w for _ in range(h)]
    for i, j in shp:
        gi[i + loci][j + locj] = choice(remcols)

    go = [[(0 if gi[i][j] == bgc else gi[i][j]) for j in range(locj, locj + ow)]
          for i in range(loci, loci + oh)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the generator confines the shape to an (h-1)x(w-1) window, so
    # at least one full edge row of I is pure background. Read bgc from that.
    bgc = None
    for row in (I[0], I[-1]):
        if len(set(row.tolist())) == 1:
            bgc = int(row[0])
            break
    if bgc is None:
        for col in (I[:, 0], I[:, -1]):
            if len(set(col.tolist())) == 1:
                bgc = int(col[0])
                break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # Reading: peel the frame one empty line at a time, cycling through the sides
    # (top, left, bottom, right), until every edge touches the object.
    # Then recolour every remaining background cell to 0 (one colour class).
    # Each peel is a CropGrid of the current canvas minus one edge line (full rectangle).
    r0, r1, c0, c1 = 0, hi - 1, 0, wi - 1  # current window in I coordinates
    changed = True
    while changed:
        changed = False
        for side in ("top", "left", "bottom", "right"):
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if side == "top" and h > 1 and np.all(I[r0, c0:c1 + 1] == bgc):
                ops.append(33); sels.append([1, 0, h - 2, w - 1])  # whole canvas minus top row
                r0 += 1; changed = True
            elif side == "left" and w > 1 and np.all(I[r0:r1 + 1, c0] == bgc):
                ops.append(33); sels.append([0, 1, h - 1, w - 2])  # whole canvas minus left col
                c0 += 1; changed = True
            elif side == "bottom" and h > 1 and np.all(I[r1, c0:c1 + 1] == bgc):
                ops.append(33); sels.append([0, 0, h - 2, w - 1])  # whole canvas minus bottom row
                r1 -= 1; changed = True
            elif side == "right" and w > 1 and np.all(I[r0:r1 + 1, c1] == bgc):
                ops.append(33); sels.append([0, 0, h - 1, w - 2])  # whole canvas minus right col
                c1 -= 1; changed = True

    cur = I[r0:r1 + 1, c0:c1 + 1]
    ho, wo = cur.shape

    # All cells of the background colour in the current canvas -> 0.
    bg_cells = [(r, c) for r in range(ho) for c in range(wo) if cur[r, c] == bgc]
    if bg_cells:
        ops.append(0); sels.append(sel_of(bg_cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task a740d043"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a740d043"
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
                                f"for task a740d043"
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
                    f"Failed to build a complete episode for task a740d043 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a740d043-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
