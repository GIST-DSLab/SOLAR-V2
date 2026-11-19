"""
ARC Task: 045e512c (RE-ARC) — LLM-generated grid_maker
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
    bgc, objc = random.sample(cols, 2)
    rem = [c for c in cols if c not in (bgc, objc)]
    ncols = random.randint(1, 8)
    return {"bgc": bgc, "objc": objc, "ccols": random.sample(rem, ncols)}


def _ncomp8(cells):
    cells = set(cells)
    if not cells:
        return 0
    seen = set(); n = 0
    for s in cells:
        if s in seen:
            continue
        n += 1; st = [s]; seen.add(s)
        while st:
            y, x = st.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in cells and q not in seen:
                        seen.add(q); st.append(q)
    return n


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, ccols=None, **kw) -> dict:
    def unifint(lo, hi):
        lo2 = lo + int(round(diff_lb * (hi - lo)))
        hi2 = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(lo2, hi2), max(lo2, hi2))
    if ccols is None:
        ccols = [c for c in range(10) if c not in (bgc, objc)][:1]
    h = unifint(11, max(11, max_h))
    w = unifint(11, max(11, max_w))
    while True:
        oh = unifint(2, min(4, (h - 2) // 3))
        ow = unifint(2, min(4, (w - 2) // 3))
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        c1 = (random.randint(0, oh - 1), 0)
        c2 = (0, random.randint(0, ow - 1))
        c3 = (oh - 1, random.randint(0, ow - 1))
        c4 = (random.randint(0, oh - 1), ow - 1)
        obj = {c1, c2, c3, c4}
        rem = list(bounds - obj)
        for _ in range(unifint(0, len(rem))):
            loc = random.choice(rem); obj.add(loc); rem.remove(loc)
        if _ncomp8(obj) == 1:
            break
    loci = random.randint(oh + 1, h - 2 * oh - 1)
    locj = random.randint(ow + 1, w - 2 * ow - 1)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    obj = [(i + loci, j + locj) for i, j in obj]
    for i, j in obj:
        gi[i, j] = objc; go[i, j] = objc
    options = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    ndirs = unifint(1, 8)
    dirs = random.sample(options, ndirs)
    dcols = [random.choice(ccols) for _ in range(ndirs)]
    rows = {loci - 2, loci + oh + 1}
    colsb = {locj - 2, locj + ow + 1}
    for (a, b), col in zip(dirs, dcols):
        ind = [(i + a * (oh + 1), j + b * (ow + 1)) for i, j in obj]
        ind = [(i, j) for i, j in ind if i in rows or j in colsb]
        if not ind:
            continue
        nd = unifint(0, len(ind) - 1)
        ind = random.sample(ind, len(ind) - nd)
        if 0 < len(ind) < len(obj):
            for i, j in ind:
                if 0 <= i < h and 0 <= j < w:
                    gi[i, j] = col
            for k in range(1, 10):
                for i, j in obj:
                    y, x = i + k * a * (oh + 1), j + k * b * (ow + 1)
                    if 0 <= y < h and 0 <= x < w:
                        go[y, x] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """The code-golf reading and the maker's reading AGREE, so the op structure
    is unchanged: the unit is the master shape (largest 8-connected same-colour
    component, size oh x ow); for each of the 8 directions the partial copy one
    step (oh+1, ow+1) away tells whether the direction is live and its colour;
    full copies are stamped outward at every multiple of the step until the
    grid edge. One Color op per stamped copy, selection = the cells that copy
    lands on (clipped to the grid), copies ordered outward from the master.
    Nothing here reads O."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cand = Counter(I.flatten().tolist())
    if examples:
        cnt = Counter()
        for ei, _ in examples:
            cnt.update(np.asarray(ei).flatten().tolist())
        bgc = max(cand, key=lambda c: (cand[c], cnt[c]))
    else:
        bgc = cand.most_common(1)[0][0]

    seen = set(); comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            col = int(I[r, c]); st = [(r, c)]; seen.add((r, c)); comp = []
            while st:
                y, x = st.pop(); comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and I[ny, nx] == col:
                            seen.add((ny, nx)); st.append((ny, nx))
            comps.append(comp)

    ops, sels = [], []
    if comps:
        best = sorted(max(comps, key=len))            # master shape
        oh = max(p[0] for p in best) - min(p[0] for p in best) + 1
        ow = max(p[1] for p in best) - min(p[1] for p in best) + 1
        sr, sc = oh + 1, ow + 1
        g = I.copy()                                  # current working grid (tracks our own ops)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                probe = [(y + dr * sr, x + dc * sc) for y, x in best]
                vals = [int(I[y, x]) for y, x in probe
                        if 0 <= y < h and 0 <= x < w and I[y, x] != bgc]
                if not vals:
                    continue                          # no indicator -> direction not live
                col = Counter(vals).most_common(1)[0][0]
                k = 1
                while True:                           # stamp outward until off-grid
                    cells = [(y + k * dr * sr, x + k * dc * sc) for y, x in best]
                    cells = [(y, x) for y, x in cells if 0 <= y < h and 0 <= x < w]
                    if not cells:
                        break
                    if any(g[y, x] != col for y, x in cells):
                        ops.append(int(col)); sels.append(sel_of(cells))
                        for y, x in cells:
                            g[y, x] = col
                    k += 1

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
                        f"num_examples+1 ({num_examples + 1}) for task 045e512c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 045e512c"
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
                                f"for task 045e512c"
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
                    f"Failed to build a complete episode for task 045e512c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"045e512c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
