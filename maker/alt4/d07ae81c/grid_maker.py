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
    """Fix the four colour roles for the whole episode:
    c1 / c2 = the two region colours, c3 = marker sitting on c1, c4 = marker sitting on c2."""
    cols = list(range(10))
    c1, c2, c3, c4 = random.sample(cols, 4)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             c1: int, c2: int, c3: int, c4: int) -> dict:
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))

    def roles(g):
        """marker colour -> region colour, exactly as the task's rule reads them:
        a marker colour is one whose cells are never 8-adjacent to each other;
        its region is the commonest colour in the union of their neighbourhoods."""
        hh = len(g)
        ww = len(g[0])
        pos = {}
        for r in range(hh):
            for q in range(ww):
                pos.setdefault(g[r][q], []).append((r, q))
        out = {}
        for col, pts in pos.items():
            s = set(pts)
            adj = False
            for (r, q) in pts:
                for dr in (-1, 0, 1):
                    for dq in (-1, 0, 1):
                        if (dr or dq) and (r + dr, q + dq) in s:
                            adj = True
                            break
                    if adj:
                        break
                if adj:
                    break
            if adj:
                continue
            nb = set()
            for (r, q) in pts:
                for dr in (-1, 0, 1):
                    for dq in (-1, 0, 1):
                        if dr or dq:
                            rr, qq = r + dr, q + dq
                            if 0 <= rr < hh and 0 <= qq < ww:
                                nb.add((rr, qq))
            cnt = {}
            for (r, q) in nb:
                cnt[g[r][q]] = cnt.get(g[r][q], 0) + 1
            out[col] = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return out

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    for _attempt in range(500):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        magiccol = 0
        gi = canvas(0, (h, w))
        ndivi = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
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
        if len(ofc1) == 0 or len(ofc2) == 0:
            continue
        mn = min(len(ofc1), len(ofc2))
        n1 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
        n2 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
        srcs1 = set()
        for k in range(n1):
            cands = totuple((ofc1 - srcs1) - mapply(neighbors, frozenset(srcs1)))
            if len(cands) == 0:
                break
            srcs1.add(choice(cands))
        srcs2 = set()
        for k in range(n2):
            cands = totuple((ofc2 - srcs2) - mapply(neighbors, frozenset(srcs2)))
            if len(cands) == 0:
                break
            srcs2.add(choice(cands))
        if len(srcs1) == 0 or len(srcs2) == 0:
            continue
        gi = fill(gi, c3, frozenset(srcs1))
        gi = fill(gi, c4, frozenset(srcs2))
        # the marker/region roles must be readable from the input alone, and distinct
        if roles(gi) != {c3: c1, c4: c2}:
            continue
        lns = mapply(lnf, frozenset(srcs1)) | mapply(lnf, frozenset(srcs2))
        ofc3 = ofc1 & lns
        ofc4 = ofc2 & lns
        go = fill(gi, c3, ofc3)
        go = fill(go, c4, ofc4)
        return {'input': gi, 'output': go}
    raise ValueError('generation failed')


def derive_operations(I, O, examples=None):
    """Rule: the grid is tiled by two kinds of region (colours R_a / R_b). A handful of
    isolated marker pixels sit on those regions, one marker colour per region kind.
    Every marker fires four diagonal rays (an X) right across the grid; each cell a ray
    crosses lights up in the marker colour belonging to *that cell's own* region.

    So the work is grouped by marker: one Color op for the part of this marker's X that
    lies in one kind of region, one for the part lying in the other kind."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    G = np.asarray(I, dtype=int)
    h, w = G.shape
    cur = G.copy()
    ops, sels = [], []

    pos = {}
    for r in range(h):
        for c in range(w):
            pos.setdefault(int(G[r, c]), []).append((r, c))

    def self_adjacent(pts):
        s = set(pts)
        for (r, c) in pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and (r + dr, c + dc) in s:
                        return True
        return False

    def neighbourhood(pts):
        nb = set()
        for (r, c) in pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            nb.add((rr, cc))
        return nb

    # marker colours: colours whose cells never touch one another (isolated pixels)
    marker_cols = [c for c, pts in sorted(pos.items()) if not self_adjacent(pts)]
    region_of = {}
    for m in marker_cols:
        cnt = Counter(int(G[r, c]) for (r, c) in neighbourhood(pos[m]))
        region_of[m] = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def beam(r, c):
        cells = [(r, c)]
        for dr, dc in ((1, 1), (-1, -1), (-1, 1), (1, -1)):
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                cells.append((rr, cc))
                rr += dr
                cc += dc
        return cells

    # every marker pixel, in reading order; each one fires its X
    markers = sorted([(r, c, m) for m in marker_cols for (r, c) in pos[m]])
    for (r, c, m) in markers:
        ray_cells = beam(r, c)
        # this marker's own region first, then the other region the same X crosses
        order = [m] + [o for o in marker_cols if o != m]
        for col in order:
            reg = region_of[col]
            sel = [(rr, cc) for (rr, cc) in ray_cells
                   if G[rr, cc] == reg and cur[rr, cc] != col]
            if not sel:
                continue
            ops.append(int(col))
            sels.append(sel_of(sel))
            for (rr, cc) in sel:
                cur[rr, cc] = col

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # full-grid rectangle: submit
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
