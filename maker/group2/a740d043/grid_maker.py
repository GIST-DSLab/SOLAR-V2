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
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background is never 0 (the generator draws it from 1..9). It is fixed for the
    # whole episode. Foreground colours do not affect the rule, so each instance
    # picks its own.
    bgc = random.choice(list(range(1, 10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(max(a + round(d * (b - a)), a), b)

    cols = list(range(1, 10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, ((h - 1) * (w - 1)) // 2)))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, len(remcols)))
    remcols = random.sample(remcols, numc)

    gi = np.full((h, w), bgc, dtype=int)
    bounds = {(i, j) for i in range(h - 1) for j in range(w - 1)}
    ch = random.choice(sorted(bounds))
    shp = {ch}
    bounds.discard(ch)
    for _ in range(nc):
        nbrs = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        nbrs.add((i + di, j + dj))
        cand = sorted((bounds - shp) & nbrs)
        if not cand:
            break
        shp.add(random.choice(cand))
    mi = min(i for i, _ in shp)
    mj = min(j for _, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, _ in shp) + 1
    ow = max(j for _, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    for (i, j) in shp:
        gi[i + loci, j + locj] = random.choice(remcols)

    mask = gi != bgc
    rows = np.where(mask.any(axis=1))[0]
    cs = np.where(mask.any(axis=0))[0]
    go = gi[rows.min():rows.max() + 1, cs.min():cs.max() + 1].copy()
    go[go == bgc] = 0
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the shape fits inside (h-1) x (w-1), so it can never fill a whole
    # row. Any row made of one colour is therefore background.
    bgc = None
    for r in range(hi):
        if np.all(I[r] == I[r, 0]):
            bgc = int(I[r, 0])
            break
    if bgc is None:
        vals, counts = np.unique(I, return_counts=True)
        bgc = int(vals[np.argmax(counts)])

    ops, sels = [], []

    # Step 1 (one colour at a time, whole grid): repaint every background-coloured
    # cell to 0 in one stroke. bgc is never 0, so this always changes the grid.
    bg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]
    ops.append(0)
    sels.append(sel_of(bg_cells))

    # Step 2: crop to the shape's bounding rectangle. The selection is the full
    # rectangle around the non-background cells, so a bbox is correct here.
    # Background is now 0, so the transparent crop leaves it as 0.
    mask = I != bgc
    rows = np.where(mask.any(axis=1))[0]
    cs = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cs.min()), int(cs.max())
    ops.append(33)
    sels.append([r0, c0, r1 - r0, c1 - c0])

    ho, wo = r1 - r0 + 1, c1 - c0 + 1
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
