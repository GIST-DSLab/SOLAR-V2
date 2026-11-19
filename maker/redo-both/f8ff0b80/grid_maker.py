"""
ARC Task: f8ff0b80 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is a structural role here: the rule (order the
    # objects by size, reflect the resulting colour column) does not depend on
    # which colours the objects have, so object colours stay free per instance.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def make_shape(size):
        base = [(i, j) for i in range(6) for j in range(6)]
        sp = random.choice(base)
        shp = {sp}
        for _ in range(size - 1):
            cand = [x for x in base
                    if x not in shp
                    and any(abs(x[0] - y[0]) <= 1 and abs(x[1] - y[1]) <= 1 for y in shp)]
            if not cand:
                break
            shp.add(random.choice(cand))
        if len(shp) != size:
            return None
        mr = min(r for r, c in shp)
        mc = min(c for r, c in shp)
        return frozenset((r - mr, c - mc) for r, c in shp)

    cols = [c for c in range(10) if c != bgc]
    hlo = min(10, max_h)
    wlo = min(10, max_w)

    best = None
    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        area = h * w
        maxobjs = max(2, min(8, area // 25))
        nobjs = unifint(diff_lb, diff_ub, (2, maxobjs))

        # DISTINCT object sizes: the output order is "sorted by size", so equal
        # sizes would make the ordering ambiguous / unlearnable.
        pool = list(range(1, max(nobjs + 1, min(13, nobjs + 6))))
        sizes = None
        for _t in range(30):
            cand = sorted(random.sample(pool, nobjs))
            if sum(cand) <= min(36, area // 4):
                sizes = cand
                break
        if sizes is None:
            sizes = list(range(1, nobjs + 1))
        while len(sizes) > 2 and sum(sizes) > min(36, area // 4):
            sizes.pop()
        if sum(sizes) > area // 4:
            continue

        inds = set((i, j) for i in range(h) for j in range(w))
        placed = []
        for s in sizes:
            shp = None
            for _t in range(10):
                shp = make_shape(s)
                if shp is not None:
                    break
            if shp is None:
                continue
            sh = max(r for r, c in shp) + 1
            sw = max(c for r, c in shp) + 1
            if sh > h or sw > w:
                continue
            cands = [ij for ij in inds if ij[0] <= h - sh and ij[1] <= w - sw]
            found = None
            for _tr in range(20):
                if not cands:
                    break
                loc = random.choice(cands)
                plcd = frozenset((loc[0] + r, loc[1] + c) for r, c in shp)
                if plcd <= inds:
                    found = plcd
                    break
            if found is None:
                continue
            placed.append((s, found))
            halo = set()
            for (r, c) in found:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        halo.add((r + dr, c + dc))
            inds = inds - found - halo

        if len(placed) < 2:
            continue

        placed.sort(key=lambda t: t[0])          # ascending size
        n = len(placed)
        # colour sequence must not be a palindrome, otherwise the reflection
        # that defines this task would be invisible
        colseq = [random.choice(cols) for _ in range(n)]
        tries = 0
        while colseq == colseq[::-1] and tries < 100:
            colseq = [random.choice(cols) for _ in range(n)]
            tries += 1
        if colseq == colseq[::-1]:
            alt = [c for c in cols if c != colseq[0]]
            if alt:
                colseq[-1] = random.choice(alt)

        gi = [[bgc for _ in range(w)] for _ in range(h)]
        for (s, cells), col in zip(placed, colseq):
            for (r, c) in cells:
                gi[r][c] = col

        gi = tuple(tuple(row) for row in gi)
        go = tuple((c,) for c in reversed(colseq))   # hmirror of the ascending column
        best = {'input': gi, 'output': go}
        break

    if best is None:
        # minimal but valid fallback instance, still built the same way
        h = max(10, hlo)
        w = max(10, wlo)
        c1 = random.choice(cols)
        c2 = random.choice([c for c in cols if c != c1])
        gi = [[bgc for _ in range(w)] for _ in range(h)]
        gi[0][0] = c1
        gi[3][3] = c2
        gi[3][4] = c2
        best = {'input': tuple(tuple(r) for r in gi),
                'output': ((c2,), (c1,))}
    return best


def derive_operations(I, O):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # --- everything below is measured from I only -------------------------
    # background: the colour the canvas was painted with before objects were
    # placed; objects cover at most a quarter of the canvas, so it is the
    # strict majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # objects: same-colour, diagonally connected components of non-background
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append((len(cells), min(cells), col))

    # reading order of the objects: smallest first (ties broken by position,
    # though the generator gives every object a distinct size)
    comps.sort(key=lambda t: (t[0], t[1]))
    asc = [t[2] for t in comps]
    n = max(1, len(asc))

    ops, sels = [], []

    # 1. shrink the canvas to the n x 1 column that will hold one cell per
    #    object (n counted from I).  Full-rectangle selection: the crop bbox.
    ops.append(33)
    sels.append([0, 0, n - 1, 0])
    cur = [int(I[i, 0]) for i in range(n)]   # transparent crop keeps these values

    # 2. write one cell per object, smallest object at the top
    for i, col in enumerate(asc):
        if cur[i] != col:
            ops.append(int(col))
            sels.append(sel_of([(i, 0)]))
            cur[i] = int(col)

    # 3. reflect the column up<->down (the mirror this task is about);
    #    selection is the whole n x 1 grid, a genuine full rectangle
    if n > 1 and cur != cur[::-1]:
        ops.append(27)
        sels.append([0, 0, n - 1, 0])

    ops.append(34)
    sels.append([0, 0, n - 1, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8ff0b80"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8ff0b80"
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
                                f"for task f8ff0b80"
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
                    f"Failed to build a complete episode for task f8ff0b80 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8ff0b80-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
