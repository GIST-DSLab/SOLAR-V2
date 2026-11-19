"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dotc is None:
        dotc = choice(remove(bgc, cols))
    lim = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(4, lim // 2))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(4, lim // 2))))
    remcols = [c for c in cols if c not in (bgc, dotc)]
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    lo = min(h, w) * 2
    gih = unifint(diff_lb, diff_ub, (lo, max(lo, max_h)))
    giw = unifint(diff_lb, diff_ub, (lo, max(lo, max_w)))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    for _ in range(60):
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                if gih - oh < 0 or giw - ow < 0:
                    break
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            return {'input': gi, 'output': go}
        maxtr = min(int(maxtr * 1.5) + 1, 500)
        gih = randint(gih, max(gih, max_h))
        giw = randint(giw, max(giw, max_w))
    raise ValueError("placement failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    exs = [np.asarray(a, dtype=int) for a, _ in (examples or [])]

    # background is shared by every input of the episode
    cnt = Counter(I.flatten().tolist())
    for a in exs:
        cnt.update(a.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    colcells = {}
    for r in range(H):
        for c in range(W):
            if I[r, c] != bgc:
                colcells.setdefault(int(I[r, c]), set()).add((r, c))

    def ncomp(cells):
        cells = set(cells); n = 0
        while cells:
            st = [cells.pop()]; n += 1
            while st:
                r, c = st.pop()
                for nb in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                    if nb in cells:
                        cells.remove(nb); st.append(nb)
        return n

    # marker colour: one marker per fragment -> #marker cells == #other colours;
    # it is the same colour in every demonstration input
    common = set(colcells)
    for a in exs:
        common &= set(np.unique(a).tolist())
    order_c = lambda pool: sorted(pool, key=lambda k: (-ncomp(colcells[k]), len(colcells[k])))
    cands = [k for k in order_c(common) if len(colcells[k]) == len(colcells) - 1]
    cands += [k for k in order_c(colcells) if k not in cands and len(colcells[k]) == len(colcells) - 1]

    def assign(dotc):
        dots = sorted(colcells[dotc])
        others = sorted(k for k in colcells if k != dotc)
        dist = lambda p, cs: min(abs(p[0]-q[0]) + abs(p[1]-q[1]) for q in cs)
        prefs = {k: sorted(dots, key=lambda p: (dist(p, colcells[k]), p)) for k in others}
        others.sort(key=lambda k: len(colcells[k]), reverse=True)
        res = {}
        def rec(i, used, occ):
            if i == len(others):
                rs = [r for r, _ in occ]; cs_ = [c for _, c in occ]
                return len(occ) == (max(rs)-min(rs)+1) * (max(cs_)-min(cs_)+1)
            k = others[i]
            for d in prefs[k]:
                if d in used:
                    continue
                rel = {(r - d[0], c - d[1]) for r, c in colcells[k]}
                if rel & occ:
                    continue
                nocc = occ | rel
                rs = [r for r, _ in nocc]; cs_ = [c for _, c in nocc]
                if max(rs)-min(rs)+1 > 4 or max(cs_)-min(cs_)+1 > 4:
                    continue
                res[k] = d
                if rec(i + 1, used | {d}, nocc):
                    return True
                del res[k]
            return False
        return (dotc, dict(res)) if rec(0, frozenset(), {(0, 0)}) else None

    found = None
    for dc_ in cands:
        found = assign(dc_)
        if found is not None:
            break
    if found is None:
        raise ValueError("137eaa0f: no consistent grouping of fragments")
    dotc, mapping = found

    frags = []
    for k, d in mapping.items():
        cells = sorted(colcells[k] | {d})
        frags.append({"color": k, "dot": d, "cells": cells})
    allrel = [(0, 0)] + [(r - f["dot"][0], c - f["dot"][1]) for f in frags for r, c in f["cells"]]
    minr = min(r for r, _ in allrel); minc = min(c for _, c in allrel)
    th = max(r for r, _ in allrel) - minr + 1
    tw = max(c for _, c in allrel) - minc + 1

    ops, sels = [], []
    # 1. cut the canvas down to the th x tw frame of the assembled picture
    #    (whole rectangle intended: the canvas is the frame).  Whatever the crop
    #    carries over from the input is overwritten by the stamps below.
    ops.append(33); sels.append([0, 0, th - 1, tw - 1])
    canvas_ = np.zeros((th, tw), dtype=int)
    for r in range(min(th, H)):
        for c in range(min(tw, W)):
            canvas_[r, c] = I[r, c]

    def dest(f):
        return [(r - f["dot"][0] - minr, c - f["dot"][1] - minc) for r, c in f["cells"]]

    # 2. copy each fragment out of the input and stamp it into the frame with its
    #    marker on the shared marker cell; biggest pieces first, pieces whose only
    #    visible cell is the marker last
    frags.sort(key=lambda f: (-sum(1 for p in f["cells"] if I[p] != 0), f["color"]))
    for f in frags:
        r0 = min(r for r, _ in f["cells"]); c0 = min(c for _, c in f["cells"])
        dr0 = r0 - f["dot"][0] - minr; dc0 = c0 - f["dot"][1] - minc
        new = canvas_.copy()
        for (r, c), (rr, cc) in zip(f["cells"], dest(f)):
            if I[r, c] != 0:
                new[rr, cc] = I[r, c]
        if np.array_equal(new, canvas_):
            continue   # the stamp would change nothing visible
        ops.append(28); sels.append(sel_of(f["cells"]))
        ops.append(30); sels.append([dr0, dc0, 0, 0])
        canvas_ = new

    # 3. a piece drawn in colour 0 cannot be copied (0 is "empty" to Copy/Paste):
    #    paint its landing cells 0 directly
    zero = []
    for f in frags:
        for (r, c), (rr, cc) in zip(f["cells"], dest(f)):
            if I[r, c] == 0 and canvas_[rr, cc] != 0:
                zero.append((rr, cc))
    if zero:
        ops.append(0); sels.append(sel_of(zero))

    ops.append(34); sels.append([0, 0, th - 1, tw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
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
                                f"for task 137eaa0f"
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
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
