"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # the cross must be non-zero so ARCLE can grab and Move it as an object
    fgc = random.choice([c for c in cols if c not in (bgc, 0)])
    dotc = random.choice([c for c in cols if c not in (bgc, fgc)])
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, dotc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None or fgc is None or dotc is None:
        bgc, fgc, dotc = random.sample([c for c in cols if c != 0], 3)
    rotf = random.choice((identity, rot90, rot180, rot270))
    lim = min(max_h, max_w) if rotf in (rot90, rot270) else None
    hmax = min(30, max_h if lim is None else lim)
    wmax = min(30, max_w if lim is None else lim)
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = unifint(diff_lb, diff_ub, (3, wmax))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = sample(totuple(inds), nn)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, fgc, hfrontier((loci, 0)) | vfrontier((0, locj)))
    gi = fill(gi, dotc, ss)
    go = fill(go, fgc, hfrontier((loci + nn, 0)) | vfrontier((0, locj + nn)))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the code-golf entry, which rolls the whole grid by n):
    the cross is ONE object, translated diagonally by n cells, where n is the
    number of dot cells. It moves away from the dots. The maker above instead
    erases two lines and paints two new ones, so nothing shows the cross moving.
    Here the cross is grabbed whole and slid. The cross lines span the grid, so
    ARCLE clips the leading ends and the trailing ends fall short; the lines
    are then extended back to the border. The dots are used up.

    Route:
      1. Color(bgc) on all dot cells: the counters disappear.
      2. Grab the whole cross (its row and its column). Slide it n cells
         vertically, then n cells horizontally: one grab, then Moves with an
         empty selection. ARCLE clips whatever leaves the grid.
      3. Color(bgc) on the part of the cross's original footprint it vacated.
      4. Color(fgc) on the cells of the moved row and column that are not
         fgc yet in the current grid, which extends the lines to the border.
    Every quantity comes from I alone; O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # frontier row: the only uniform row (every row crosses the vertical line)
    hrow = next(r for r in range(hi) if np.all(I[r, :] == I[r, 0]))
    fgc = int(I[hrow, 0])
    vcol = next(c for c in range(wi) if np.all(I[:, c] == fgc))
    # background outnumbers dots by generator construction
    rest = Counter(int(v) for v in I.flatten().tolist() if v != fgc)
    bgc = rest.most_common(1)[0][0]
    dots = [(r, c) for r in range(hi) for c in range(wi)
            if I[r, c] != fgc and I[r, c] != bgc]
    nn = len(dots)

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    mr = sum(r for r, _ in dots) / nn
    mc = sum(c for _, c in dots) / nn
    di = sign(hrow - mr)
    dj = sign(vcol - mc)

    g = I.copy()
    # 1. the dots disappear
    ops.append(bgc); sels.append(sel_of(dots))
    for r, c in dots:
        g[r, c] = bgc

    # 2. slide the cross as one object: one grab, then empty-selection moves
    cross = sorted({(hrow, c) for c in range(wi)} | {(r, vcol) for r in range(hi)})
    cur = list(cross)
    steps = [(21 if di > 0 else 20, di, 0)] * nn + [(22 if dj > 0 else 23, 0, dj)] * nn
    first = True
    for op, dr, dc in steps:
        ops.append(op)
        sels.append(sel_of(cur if first else []))
        first = False
        cur = [(r + dr, c + dc) for r, c in cur]
    cur = [(r, c) for r, c in cur if 0 <= r < hi and 0 <= c < wi]
    for r, c in cross:
        g[r, c] = 0
    for r, c in cur:
        g[r, c] = fgc

    # 3. repair the vacated original footprint
    hole = sorted(set(cross) - set(cur))
    if bgc != 0 and hole:
        ops.append(bgc); sels.append(sel_of(hole))
    for r, c in hole:
        g[r, c] = bgc

    # 4. extend the moved frontier lines back to the grid border
    nr, nc = hrow + di * nn, vcol + dj * nn
    line = [(nr, c) for c in range(wi)] + [(r, nc) for r in range(hi)]
    ext = sorted({(r, c) for r, c in line if g[r, c] != fgc})
    if ext:
        ops.append(fgc); sels.append(sel_of(ext))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])  # whole grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
