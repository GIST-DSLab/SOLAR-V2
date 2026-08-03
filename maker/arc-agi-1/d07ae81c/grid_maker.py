"""
ARC Task: d07ae81c (RE-ARC) — LLM-generated grid_maker
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
    c1, c2, c3, c4 = random.sample(cols, 4)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             c1: int, c2: int, c3: int, c4: int) -> dict:
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    magiccol = 0
    gi = canvas(0, (h, w))
    ndivi = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    for k in range(ndivi):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: min(shape(o)) > 3 and max(shape(o)) > 4)
        objs = sfilter(objs, lambda o: height(o) * width(o) == len(o))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        if choice((True, False)):
            loci = randint(uppermost(obj) + 2, lowermost(obj) - 1)
            newobj = backdrop(frozenset({(loci, leftmost(obj)), lrcorner(obj)}))
        else:
            locj = randint(leftmost(obj) + 2, rightmost(obj) - 1)
            newobj = backdrop(frozenset({(uppermost(obj), locj), lrcorner(obj)}))
        magiccol += 1
        gi = fill(gi, magiccol, newobj)
    objs = objects(gi, T, F, F)
    for ii, obj in enumerate(objs):
        col = c1 if ii == 0 else (c2 if ii == 1 else choice((c1, c2)))
        gi = fill(gi, col, toindices(obj))
    ofc1 = ofcolor(gi, c1)
    ofc2 = ofcolor(gi, c2)
    mn = min(len(ofc1), len(ofc2))
    n1 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    n2 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    srcs1 = set()
    for k in range(n1):
        cands = totuple((ofc1 - srcs1) - mapply(neighbors, srcs1))
        if len(cands) == 0:
            break
        srcs1.add(choice(cands))
    srcs2 = set()
    for k in range(n2):
        cands = totuple((ofc2 - srcs2) - mapply(neighbors, srcs2))
        if len(cands) == 0:
            break
        srcs2.add(choice(cands))
    gi = fill(gi, c3, srcs1)
    gi = fill(gi, c4, srcs2)
    lns = mapply(lnf, srcs1) | mapply(lnf, srcs2)
    ofc3 = ofc1 & lns
    ofc4 = ofc2 & lns
    go = fill(gi, c3, ofc3)
    go = fill(go, c4, ofc4)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def nbrs(r, c):
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.append((rr, cc))
        return out

    # --- 1. group I's cells by color -------------------------------------
    cells_by_color = {}
    for r in range(h):
        for c in range(w):
            cells_by_color.setdefault(int(I[r, c]), []).append((r, c))

    # --- 2. marker colors: colors whose every cell is isolated (no same-
    #        color 8-neighbour).  Region colors form big blocks, so they fail.
    marker_colors = []
    for col, cells in cells_by_color.items():
        s = set(cells)
        isolated = True
        for (r, c) in cells:
            if any(p in s for p in nbrs(r, c)):
                isolated = False
                break
        if isolated:
            marker_colors.append(col)

    # --- 3. pair each marker color with the region color it sits in -------
    #        (majority color of the marker cells' neighbourhood)
    paint = {}
    for m in marker_colors:
        nb = set()
        for (r, c) in cells_by_color[m]:
            nb.update(nbrs(r, c))
        cnt = Counter(int(I[r, c]) for (r, c) in nb if int(I[r, c]) not in marker_colors)
        if not cnt:
            continue
        region = cnt.most_common(1)[0][0]
        paint[region] = m

    # --- 4. shoot 4 diagonal rays from each marker; recolor each region
    #        cell the ray crosses to that region's marker color.
    #        One marker = one object; its four rays are painted back to back,
    #        each ray walked outward from the marker.
    painted = set()
    markers = sorted([(r, c) for m in marker_colors for (r, c) in cells_by_color[m]])
    for (mr, mc) in markers:
        for (dr, dc) in ((1, 1), (-1, -1), (-1, 1), (1, -1)):
            r, c = mr + dr, mc + dc
            while 0 <= r < h and 0 <= c < w:
                col = int(I[r, c])
                if col in paint and (r, c) not in painted:
                    painted.add((r, c))
                    ops.append(int(paint[col]))
                    sels.append([r, c, 0, 0])
                r += dr
                c += dc

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
                        f"num_examples+1 ({num_examples + 1}) for task d07ae81c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d07ae81c"
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
                                f"for task d07ae81c"
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
                    f"Failed to build a complete episode for task d07ae81c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d07ae81c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
