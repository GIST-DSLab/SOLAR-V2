"""
ARC Task: 484b58aa (RE-ARC) — LLM-generated grid_maker
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
    import random
    cols = list(range(10))
    noisec = random.choice(cols)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"noisec": noisec, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=0, ccols=None) -> dict:
    # Same construction as generate_484b58aa, but with episode-fixed colours and
    # max_h / max_w instead of the hardcoded 30, plus a self-check that the
    # occlusion is actually recoverable from the input alone.
    cols = interval(0, 10, 1)
    if ccols is None:
        remcols_ = remove(noisec, cols)
        ccols = sample(remcols_, unifint(diff_lb, diff_ub, (2, 9)))
    ccols = tuple(ccols)

    hmax = max(10, min(30, max_h))
    wmax = max(10, min(30, max_w))

    def _restore(g):
        """Recover the clean wallpaper from a noise-occluded grid, using g only."""
        hh = len(g)
        ww = len(g[0])

        def ncomps(col):
            seen = [[False] * ww for _ in range(hh)]
            n = 0
            for r in range(hh):
                for c in range(ww):
                    if g[r][c] == col and not seen[r][c]:
                        n += 1
                        st = [(r, c)]
                        seen[r][c] = True
                        while st:
                            a, b = st.pop()
                            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                na, nb = a + da, b + db
                                if 0 <= na < hh and 0 <= nb < ww and not seen[na][nb] and g[na][nb] == col:
                                    seen[na][nb] = True
                                    st.append((na, nb))
            return n

        pal = sorted({v for row in g for v in row})
        info = {c: (ncomps(c), sum(row.count(c) for row in g)) for c in pal}
        mn = min(v[0] for v in info.values())
        nc = min([c for c in pal if info[c][0] == mn], key=lambda c: (info[c][1], c))

        rws = [r for r in range(hh) if nc not in g[r]]
        cls = [c for c in range(ww) if all(g[r][c] != nc for r in range(hh))]

        ph = ww
        if rws:
            for p in range(1, ww):
                if all(g[r][c] == g[r][c + p] for r in rws for c in range(ww - p)):
                    ph = p
                    break
        pv = hh
        if cls:
            for p in range(1, hh):
                if all(g[r][c] == g[r + p][c] for c in cls for r in range(hh - p)):
                    pv = p
                    break

        R = max(hh // max(pv, 1), ww // max(ph, 1)) + 2
        offs = [(i, j) for i in range(-R, R + 1) for j in range(-R, R + 1) if (i, j) != (0, 0)]
        offs.sort(key=lambda t: (max(abs(t[0]), abs(t[1])), abs(t[0]) + abs(t[1])))

        out = [list(row) for row in g]
        for r in range(hh):
            for c in range(ww):
                if g[r][c] == nc:
                    for i, j in offs:
                        rr = r + i * pv
                        cc = c + j * ph
                        if 0 <= rr < hh and 0 <= cc < ww and g[rr][cc] != nc:
                            out[r][c] = g[rr][cc]
                            break
        return nc, pv, ph, tuple(tuple(x) for x in out)

    gi, go = None, None
    for _attempt in range(80):
        h = unifint(diff_lb, diff_ub, (10, hmax))
        w = unifint(diff_lb, diff_ub, (10, wmax))
        hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
        wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(choice(ccols), ij) for ij in pinds})
        go = canvas(-1, (h, w))
        locs = set()
        ofs = randint(1, hp - 1)
        for a in range(2 * (h // hp + 1)):
            for b in range(w // wp + 1):
                loci = hp * a - ofs * b
                locj = wp * b
                locs.add((loci, locj))
                go = paint(go, shift(pobj, (loci, locj)))
        numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
        gi = tuple(e for e in go)
        places = apply(lbind(shift, pinds), locs)
        succ = 0
        tr = 0
        maxtr = 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph_ = randint(2, 6)
            pw_ = randint(2, 6)
            loci = randint(0, h - ph_)
            locj = randint(0, w - pw_)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph_ - 1, locj + pw_ - 1)}))
            gi2 = fill(gi, noisec, ptch)
            if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
                if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
                   len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                    succ += 1
                    gi = gi2
        rotf = choice((identity, rot90, rot180, rot270))
        gi = rotf(gi)
        go = rotf(go)
        if gi == go:
            continue
        try:
            if _restore(gi)[3] == go:
                return {'input': gi, 'output': go}
        except Exception:
            continue
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: the grid is a periodic wallpaper, partly covered by solid patches of a
    single 'noise' colour (the colour forming the fewest connected components,
    tie-broken by fewest cells).  Restore each covered cell by copying the colour
    found one wallpaper period away.  Everything below is measured from I only;
    O supplies nothing but the final canvas size.
    """
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    A = np.asarray(I, dtype=int)
    h, w = A.shape
    ho, wo = np.asarray(O, dtype=int).shape
    g = [[int(v) for v in row] for row in A]

    ops, sels = [], []

    # ---- 1. identify the noise colour (fewest 4-connected components) ----
    def ncomps(col):
        seen = [[False] * w for _ in range(h)]
        n = 0
        for r in range(h):
            for c in range(w):
                if g[r][c] == col and not seen[r][c]:
                    n += 1
                    st = [(r, c)]
                    seen[r][c] = True
                    while st:
                        a, b = st.pop()
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and not seen[na][nb] and g[na][nb] == col:
                                seen[na][nb] = True
                                st.append((na, nb))
        return n

    pal = sorted({v for row in g for v in row})
    info = {c: (ncomps(c), sum(row.count(c) for row in g)) for c in pal}
    mn = min(v[0] for v in info.values())
    noisec = min([c for c in pal if info[c][0] == mn], key=lambda c: (info[c][1], c))

    # ---- 2. measure the wallpaper periods on the uncovered rows / columns ----
    rws = [r for r in range(h) if noisec not in g[r]]
    cls = [c for c in range(w) if all(g[r][c] != noisec for r in range(h))]

    ph = w
    if rws:
        for p in range(1, w):
            if all(g[r][c] == g[r][c + p] for r in rws for c in range(w - p)):
                ph = p
                break
    pv = h
    if cls:
        for p in range(1, h):
            if all(g[r][c] == g[r + p][c] for c in cls for r in range(h - p)):
                pv = p
                break

    # ---- 3. for every covered cell, read its colour from a periodic copy ----
    R = max(h // max(pv, 1), w // max(ph, 1)) + 2
    offs = [(i, j) for i in range(-R, R + 1) for j in range(-R, R + 1) if (i, j) != (0, 0)]
    offs.sort(key=lambda t: (max(abs(t[0]), abs(t[1])), abs(t[0]) + abs(t[1])))

    target = {}
    for r in range(h):
        for c in range(w):
            if g[r][c] != noisec:
                continue
            for i, j in offs:
                rr = r + i * pv
                cc = c + j * ph
                if 0 <= rr < h and 0 <= cc < w and g[rr][cc] != noisec:
                    target[(r, c)] = g[rr][cc]
                    break

    # ---- 4. repaint patch by patch (one Color op per colour inside a patch) ----
    seen = [[False] * w for _ in range(h)]
    patches = []
    for r in range(h):
        for c in range(w):
            if g[r][c] == noisec and not seen[r][c]:
                comp = []
                st = [(r, c)]
                seen[r][c] = True
                while st:
                    a, b = st.pop()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na][nb] and g[na][nb] == noisec:
                            seen[na][nb] = True
                            st.append((na, nb))
                comp.sort()
                patches.append(comp)

    for comp in patches:
        order = []
        groups = {}
        for cell in comp:
            col = target.get(cell)
            if col is None or col == noisec:
                continue          # nothing recoverable / already correct: skip
            if col not in groups:
                groups[col] = []
                order.append(col)
            groups[col].append(cell)
        for col in order:
            ops.append(int(col))
            sels.append(sel_of(groups[col]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # bbox: Submit covers the whole canvas
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
                        f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
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
                                f"for task 484b58aa"
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
                    f"Failed to build a complete episode for task 484b58aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"484b58aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
