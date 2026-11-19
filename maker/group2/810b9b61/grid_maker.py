"""
ARC Task: 810b9b61 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 6)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def box_of(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        out = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    out.add((r, c))
        return frozenset(out)

    def normalize(cells):
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    def shape_of(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def dmirror(cells):
        return frozenset((c, r) for r, c in cells)

    def cmirror(cells):
        h, w = shape_of(cells)
        n = normalize(cells)
        return frozenset((w - 1 - c, h - 1 - r) for r, c in n)

    def vmirror(cells):
        n = normalize(cells)
        _, w = shape_of(cells)
        return frozenset((r, w - 1 - c) for r, c in n)

    def hmirror(cells):
        n = normalize(cells)
        h, _ = shape_of(cells)
        return frozenset((h - 1 - r, c) for r, c in n)

    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        rem = [c for c in cols if c != bgc]
        ccols = random.sample(rem, random.randint(1, 6))

    hub = min(30, max_h)
    wub = min(30, max_w)
    h = unifint(diff_lb, diff_ub, (min(10, hub), hub))
    w = unifint(diff_lb, diff_ub, (min(10, wub), wub))
    nobjs = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 10)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    mfs = [lambda x: x, dmirror, cmirror, vmirror, hmirror]
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 5)
        ow = random.randint(3, 5)
        if oh > h or ow > w:
            continue
        obj = box_of([(0, 0), (oh - 1, ow - 1)])
        for fn in random.sample(mfs, random.choice((1, 2))):
            obj = normalize(fn(obj))
        oh, ow = shape_of(obj)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(cands)
        plcd = frozenset((r + li, c + lj) for r, c in obj)
        if random.choice((True, False)):
            k = unifint(diff_lb, diff_ub, (1, len(plcd) - 1))
            inobj = frozenset(random.sample(sorted(plcd), k))
        else:
            inobj = plcd
        if inobj.issubset(inds):
            succ += 1
            nb = set()
            for r, c in inobj:
                nb |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            inds = (inds - inobj) - nb
            col = random.choice(ccols)
            hh, ww = shape_of(inobj)
            isframe = box_of(inobj) == inobj and min(hh, ww) > 2
            for r, c in inobj:
                gi[r][c] = col
                go[r][c] = 3 if isframe else col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour dominating the inputs (objects are sparse outlines)
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    present = set(I.flatten().tolist())
    bgc = max(present, key=lambda c: cnt[c])

    # find 4-connected single-colour objects; keep the hollow rectangular frames
    seen = np.zeros((hi, wi), dtype=bool)
    frames_by_color = {}
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            color = int(I[r, c])
            comp = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == color:
                        seen[ny, nx] = True
                        dq.append((ny, nx))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if min(r1 - r0 + 1, c1 - c0 + 1) <= 2:
                continue
            perim = {(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                     if y in (r0, r1) or x in (c0, c1)}
            if set(comp) == perim:
                frames_by_color.setdefault(color, []).extend(comp)

    # unit of work: one source colour at a time — every frame drawn in that
    # colour is recoloured to 3 in a single Color3 stroke
    ops, sels = [], []
    for color in sorted(frames_by_color):
        ops.append(3)
        sels.append(sel_of(sorted(frames_by_color[color])))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
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
                                f"for task 810b9b61"
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
                    f"Failed to build a complete episode for task 810b9b61 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"810b9b61-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
