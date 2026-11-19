"""
ARC Task: 447fd412 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(b, max(a, round(a + (b - a) * d)))


def sample_colors(num_examples=None) -> dict:
    bgc, indic, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, indic=1, mainc=2) -> dict:
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (1, 4))
    ow = _unifint(diff_lb, diff_ub, (1, 4))
    if oh * ow < 3:
        if random.choice((True, False)):
            oh = _unifint(diff_lb, diff_ub, (3, 4))
        else:
            ow = _unifint(diff_lb, diff_ub, (3, 4))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}

    def nbrs(c):
        i, j = c
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    ncells = _unifint(diff_lb, diff_ub, (3, oh * ow))
    obj = {random.choice(sorted(bounds))}
    for _ in range(ncells - 1):
        nb = set()
        for c in obj:
            nb |= nbrs(c)
        obj.add(random.choice(sorted((bounds - obj) & nb)))
    mi = min(i for i, j in obj)
    mj = min(j for i, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    objt = sorted(obj)
    kk = len(obj)
    nindic = random.randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
    indicobj = set(random.sample(objt, nindic))
    colobj = {c: (indic if c in indicobj else mainc) for c in obj}
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for (i, j), col in colobj.items():
        gi[i + loci, j + locj] = col
        go[i + loci, j + locj] = col
    plcd = {(i + loci, j + locj) for i, j in obj}
    nb = set()
    for c in plcd:
        nb |= nbrs(c)
    inds = {(i, j) for i in range(h) for j in range(w) if gi[i, j] == bgc} - nb
    noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd)))))
    tr, succ = 0, 0
    while succ < noccs and tr < 5 * noccs:
        tr += 1
        hi_f = min(5, min(h, w) // max(oh, ow) - 1)
        if hi_f < 1:
            break
        fac = random.randint(1, hi_f)
        outobj = {}
        for (i, j), col in colobj.items():
            for a in range(fac):
                for b in range(fac):
                    outobj[(i * fac + a, j * fac + b)] = col
        hh, ww = oh * fac, ow * fac
        cands = [ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        outp = {(i + li, j + lj): col for (i, j), col in outobj.items()
                if 0 <= i + li < h and 0 <= j + lj < w}
        if set(outp).issubset(inds):
            succ += 1
            inp = [c for c, col in outp.items() if col == indic]
            nb = set()
            for c in inp:
                nb |= nbrs(c)
            inds = (inds - set(outp)) - nb
            for (i, j), col in outp.items():
                if col == indic:
                    gi[i, j] = col
                go[i, j] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading taken from the golf entry (it differs from the maker in how occurrences
    are found; the painting itself is the same):
      The template's indicator part works as a stamp key. Go through the stray
      indicator cells in raster order. The first one not yet used is the top-left
      indicator cell of an occurrence. Try the largest scale first: the occurrence
      is the biggest upscaled copy of the indicator pattern whose cells are all
      still unused. Mark those cells as used, then paint the matching upscaled
      body cells in the body colour (one Color op per occurrence).
    Everything is measured from I alone.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected non-background components
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                                    and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                st.append((ny, nx))
                comps.append(cells)
    if not comps:
        return [34], [[0, 0, h - 1, w - 1]]

    # template = multicolour component; indicator = colour that also appears elsewhere
    ti = max(range(len(comps)), key=lambda k: len({int(I[r, c]) for r, c in comps[k]}))
    tmpl = comps[ti]
    tcols = Counter(int(I[r, c]) for r, c in tmpl)
    outside = Counter(int(I[r, c]) for k, cs in enumerate(comps) if k != ti for r, c in cs)
    if len(tcols) < 2 or not outside:
        return [34], [[0, 0, h - 1, w - 1]]
    indic = outside.most_common(1)[0][0]
    if indic not in tcols:
        return [34], [[0, 0, h - 1, w - 1]]
    mainc = max((c for c in tcols if c != indic), key=lambda c: tcols[c])

    tset = set(tmpl)
    ind_cells = sorted((r, c) for r, c in tmpl if I[r, c] == indic)
    body_cells = [(r, c) for r, c in tmpl if I[r, c] == mainc]
    ar, ac = ind_cells[0]  # raster-first indicator cell = key anchor
    ind_off = [(r - ar, c - ac) for r, c in ind_cells]
    body_off = [(r - ar, c - ac) for r, c in body_cells]

    def scaled(offs, fac, r0, c0):
        return [(r0 + a * fac + da, c0 + b * fac + db)
                for a, b in offs for da in range(fac) for db in range(fac)]

    remaining = {(r, c) for r in range(h) for c in range(w)
                 if I[r, c] == indic and (r, c) not in tset}
    while remaining:
        r0, c0 = min(remaining)
        hit = None
        for fac in range(5, 0, -1):          # largest scale first
            cells = scaled(ind_off, fac, r0, c0)
            if set(cells) <= remaining:
                hit = (fac, cells)
                break
        if hit is None:
            remaining.discard((r0, c0))
            continue
        fac, cells = hit
        remaining -= set(cells)              # this key is used up
        paint = sorted({(r, c) for r, c in scaled(body_off, fac, r0, c0)
                        if 0 <= r < h and 0 <= c < w})
        if paint and any(I[r, c] != mainc for r, c in paint):
            ops.append(int(mainc))
            sels.append(sel_of(paint))       # the body of this one occurrence

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
                        f"num_examples+1 ({num_examples + 1}) for task 447fd412"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 447fd412"
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
                                f"for task 447fd412"
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
                    f"Failed to build a complete episode for task 447fd412 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"447fd412-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
