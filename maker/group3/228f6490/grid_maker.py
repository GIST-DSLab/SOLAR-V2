"""
ARC Task: 228f6490 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kwargs) -> dict:
    from random import randint, choice
    import random as _r

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = _r.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    def dneigh(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def mapdn(s):
        out = set()
        for p in s:
            out |= dneigh(p)
        return out

    def normalize(s):
        mr = min(r for r, _ in s)
        mc = min(c for _, c in s)
        return frozenset((r - mr, c - mc) for r, c in s)

    def shift(s, d):
        return frozenset((r + d[0], c + d[1]) for r, c in s)

    def rect(hh, ww):
        return frozenset((i, j) for i in range(hh) for j in range(ww))

    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    if sqc is None:
        sqc = choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c != bgc and c != sqc]

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    def fill(g, col, cells):
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                g[r][c] = col

    inds = set(rect(h, w))
    forbidden = []
    succ = 0
    tr = 0
    maxtr = 5 * nsq
    while tr < maxtr and succ < nsq:
        tr += 1
        oh = randint(3, 6)
        ow = randint(3, 6)
        bd = rect(oh, ow)
        bounds = shift(rect(oh - 2, ow - 2), (1, 1))
        obj = {choice(sorted(bounds))}
        ncells = randint(1, (oh - 2) * (ow - 2))
        for _ in range(ncells - 1):
            cand = sorted((set(bounds) - obj) & mapdn(obj))
            if not cand:
                break
            obj.add(choice(cand))
        sqcands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not sqcands:
            continue
        loc = choice(sorted(sqcands))
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            ooh = max(r for r, _ in inobjn) + 1
            oow = max(c for _, c in inobjn) + 1
            cands2 = [ij for ij in inds if ij[0] <= h - ooh and ij[1] <= w - oow]
            if not cands2:
                continue
            loc2 = choice(sorted(cands2))
            inobjplcd = shift(inobjn, loc2)
            bdnorm = frozenset(bd - frozenset(obj))
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (set(bdplcd) | set(inobjplcd))) - mapdn(inobjplcd)
                col = choice(remcols)
                oplcd = shift(frozenset(obj), loc)
                fill(gi, sqc, bdplcd - oplcd)
                fill(go, sqc, bdplcd)
                fill(go, col, oplcd)
                fill(gi, col, inobjplcd)
    nremobjs = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
    succ = 0
    tr = 0
    maxtr = 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        oh = randint(1, 4)
        ow = randint(1, 4)
        bounds = rect(oh, ow)
        obj = {choice(sorted(bounds))}
        ncells = randint(1, oh * ow)
        for _ in range(ncells - 1):
            cand = sorted((set(bounds) - obj) & mapdn(obj))
            if not cand:
                break
            obj.add(choice(cand))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = choice(sorted(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - set(plcd)) - mapdn(plcd)
            col = choice(remcols)
            fill(gi, col, plcd)
            fill(go, col, plcd)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the canvas colour the generator paints first (clear majority)
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # 8-connected single-colour components of I
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            col = int(I[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(I[ny, nx]) == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append((col, cells))

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def borders(cells):
        return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)

    def ring_colour(cells):
        s = set(cells)
        nb = set()
        for (y, x) in cells:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (dy or dx) and 0 <= ny < h and 0 <= nx < w and (ny, nx) not in s:
                        nb.add((ny, nx))
        if not nb:
            return None
        return Counter(int(I[p]) for p in nb).most_common(1)[0][0]

    holes = [cells for (col, cells) in comps if col == bgc and not borders(cells)]
    rings = [(cells, ring_colour(cells)) for cells in holes]
    rings = [(cells, rc) for cells, rc in rings if rc is not None]
    if rings:
        sqc = Counter(rc for _, rc in rings).most_common(1)[0][0]
        boxholes = [cells for cells, rc in rings if rc == sqc]
        loose = [(col, cells) for (col, cells) in comps if col != bgc]

        plan = []
        for hole in sorted(boxholes, key=lambda cs: min(cs)):
            hn = norm(hole)
            for (col, cells) in loose:
                if len(cells) == len(hole) and norm(cells) == hn:
                    plan.append((sorted(hole), sorted(cells), int(col)))
                    break

        # Unit of work: the BOX. First every box's empty pocket is flooded with
        # the colour of the loose piece whose silhouette matches it (one seed
        # per pocket - the pocket is a single region walled in by the box) ...
        for hole, src, col in plan:
            ops.append(10 + col)
            sels.append([hole[0][0], hole[0][1], 0, 0])
        # ... then, the pieces having been "used up", each loose piece is wiped
        # back to the background as a whole object.
        for hole, src, col in plan:
            ops.append(bgc)
            sels.append(sel_of(src))

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
                        f"num_examples+1 ({num_examples + 1}) for task 228f6490"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 228f6490"
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
                                f"for task 228f6490"
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
                    f"Failed to build a complete episode for task 228f6490 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"228f6490-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
