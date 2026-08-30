"""
ARC Task: e8dc4411 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """bgc / objc (the big symmetric shape) / remc (the marker + the stamped copies)
    are all sampled randomly by the original generator -> fix them for the episode.
    The only discrete structural variant is the final rotation, which decides WHICH
    of the four diagonal directions the copies march along -> plan it per instance."""
    cols = list(range(10))
    bgc, objc, remc = random.sample(cols, 3)

    variants = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "objc": objc, "remc": remc, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, objc=None, remc=None, rot=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or objc is None or remc is None:
        bgc, objc, remc = sample(cols, 3)
    if rot is None:
        rot = randint(0, 3)

    hmax = max(9, int(max_h))
    wmax = max(9, int(max_w))
    h = unifint(diff_lb, diff_ub, (9, hmax))
    w = unifint(diff_lb, diff_ub, (9, wmax))
    d = unifint(diff_lb, diff_ub, (3, min(h, w) // 2 - 1))

    c = canvas(bgc, (d, d))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= d // 2 and ij[1] >= d // 2)
    ncd = unifint(diff_lb, diff_ub, (1, len(inds) // 2))
    nc = choice((ncd, len(inds) - ncd))
    nc = min(max(2, nc), len(inds) - 1)
    cells = sample(totuple(inds), nc)
    cells = set(cells) | {choice(((d // 2, d // 2), (d // 2, d // 2 - 1)))}
    cells = cells | {(jj, ii) for ii, jj in cells}
    for k in range(4):
        c = fill(c, objc, cells)
        c = rot90(c)
    while palette(toobject(box(asindices(c)), c)) == frozenset({bgc}) and height(c) > 3:
        c = trim(c)

    obj = ofcolor(c, objc)
    od = height(obj)
    loci = randint(1, h - 2 * od)
    locj = randint(1, w - 2 * od)
    obj = shift(obj, (loci, locj))
    bd = backdrop(obj)
    p = 0
    while len(shift(obj, (p, p)) & bd) > 0:
        p += 1
    obj2 = shift(obj, (p, p))
    nbhs = mapply(neighbors, obj)
    while len(obj2 & nbhs) == 0:
        nbhs = mapply(neighbors, nbhs)
    indic = obj2 & nbhs

    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    gi = fill(gi, remc, indic)
    go = tuple(e for e in gi)
    for k in range(30):
        newg = fill(go, remc, shift(obj, (p * (k + 1), p * (k + 1))))
        if newg == go:
            break
        go = newg

    rotf = (identity, rot90, rot180, rot270)[rot % 4]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """Rule (read entirely from I):
       - one big 4-fold-symmetric shape (colour objc, the larger colour class)
       - one small marker (colour remc) sitting diagonally off one of its corners
       - p = smallest p>=1 for which the shape shifted by (p,p) no longer touches
         the shape's own bounding box  (this is the generator's step size)
       - the diagonal direction is the one whose p-step copy of the shape covers
         the marker
       - then stamp copies of the shape, in the marker colour, at k*p*dir for
         k = 1, 2, 3, ... until the copy has walked off the grid.
       O is never inspected."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # --- background: the colour the canvas was painted with (dominant by far) ---
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    classes = {}
    for r in range(h):
        for c in range(w):
            v = int(I[r, c])
            if v != bgc:
                classes.setdefault(v, []).append((r, c))

    ops, sels = [], []
    if len(classes) < 2:
        ops.append(34); sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
        return ops, sels

    # larger colour class = the shape; the other = the marker
    objc = max(classes, key=lambda k: (len(classes[k]), -k))
    remc = min([k for k in classes if k != objc], key=lambda k: (len(classes[k]), k))
    obj = set(classes[objc])
    mark = set(classes[remc])

    rmin = min(r for r, _ in obj); rmax = max(r for r, _ in obj)
    cmin = min(c for _, c in obj); cmax = max(c for _, c in obj)

    # --- step size p: first diagonal shift that clears the shape's own bbox ---
    span = max(rmax - rmin, cmax - cmin) + 2
    p = 1
    while p <= span:
        if not any(rmin <= r + p <= rmax and cmin <= c + p <= cmax for r, c in obj):
            break
        p += 1

    # --- direction: the diagonal whose first copy lands on the marker ---
    best_dir, best_hit = (1, 1), -1
    for dr in (-1, 1):
        for dc in (-1, 1):
            hit = len({(r + p * dr, c + p * dc) for r, c in obj} & mark)
            if hit > best_hit:
                best_hit, best_dir = hit, (dr, dc)
    dr, dc = best_dir

    # --- stamp one copy of the shape per step, in the marker colour ---
    cur = I.copy()
    for k in range(1, max(h, w) + 1):
        cells = [(r + k * p * dr, c + k * p * dc) for (r, c) in sorted(obj)]
        cells = [(r, c) for (r, c) in cells if 0 <= r < h and 0 <= c < w]
        if not cells:
            break                                   # copy has left the grid
        if all(cur[r, c] == remc for (r, c) in cells):
            continue                                # would change nothing
        ops.append(int(remc))
        sels.append(sel_of(cells))                  # exact cells of this stamp
        for (r, c) in cells:
            cur[r, c] = remc

    # Submit: selection is the whole grid rectangle
    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task e8dc4411"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e8dc4411"
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
                                f"for task e8dc4411"
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
                    f"Failed to build a complete episode for task e8dc4411 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e8dc4411-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
