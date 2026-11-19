"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, mainc, markerc = random.sample(cols, 3)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, mainc=None, markerc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or mainc is None or markerc is None:
        bgc, mainc, markerc = sample(cols, 3)
    hlo = min(10, max_h)
    wlo = min(10, max_w)
    while True:
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        remcols = difference(cols, (bgc, mainc, markerc))
        ncols = unifint(diff_lb, diff_ub, (1, len(remcols)))
        ccols = sample(remcols, ncols)
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        if oh > h or ow > w:
            continue
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            cands = totuple((bounds - obj) & mapply(neighbors, obj))
            if len(cands) == 0:
                break
            obj.add(choice(cands))
        if len(obj) < 4:
            continue
        ncells = len(obj)
        obj = normalize(obj)
        oh, ow = shape(obj)
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(remcols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(obj) != obj:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            ohh, oww = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - ohh and ij[1] <= w - oww)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Reading (per partial object, as the golf entry carves it):
      - 8-connected non-background components of I; the largest one is the
        TEMPLATE (main colour + marker + completion colours).
      - every other component is (a piece of) a PARTIAL copy; their colours
        form the partial palette {mainc, markerc}.
      - for each partial component (raster order of its first cell), find the
        mirror/rotation variant of the template whose partial-palette sub-shape
        lies over it in I, and paint that variant's completion cells there,
        one Color op per completion colour. Each copy is completed once.
    Selections are named from I alone: "the completion cells of the template
    variant laid over this partial object".
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = [0, 0, h - 1, w - 1]

    # background = the canvas colour (sparse objects -> majority colour of I)
    icnt = Counter(I.flatten().tolist())
    bgc = max(icnt, key=lambda c: icnt[c])

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            stack.append((na, nb))
            comps.append(sorted(cells))
    if len(comps) < 2:
        ops.append(34); sels.append(submit)
        return ops, sels

    tidx = max(range(len(comps)), key=lambda k: len(comps[k]))
    partials = [cc for k, cc in enumerate(comps) if k != tidx]
    palette = {int(I[r, c]) for cc in partials for (r, c) in cc}
    template = [(int(I[r, c]), (r, c)) for (r, c) in comps[tidx]]

    def norm(obj):
        mi = min(p[0] for _, p in obj)
        mj = min(p[1] for _, p in obj)
        return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in obj)

    fs = [lambda i, j: (i, j), lambda i, j: (j, i), lambda i, j: (-j, -i),
          lambda i, j: (-i, j), lambda i, j: (i, -j)]
    variants, vseen = [], set()
    for f1 in fs:
        for f2 in fs:
            t = norm([(v, f1(*f2(i, j))) for v, (i, j) in template])
            if t not in vseen:
                vseen.add(t)
                variants.append(t)

    done = set()
    G = I.copy()
    for comp in sorted(partials, key=lambda cc: cc[0]):
        r0, c0 = comp[0]
        v0 = int(I[r0, c0])
        for var in variants:
            sub = [(v, p) for v, p in var if v in palette]
            fill = [(v, p) for v, p in var if v not in palette]
            if not sub or not fill:
                continue
            for v, (a, b) in sub:
                if v != v0:
                    continue
                oi, oj = r0 - a, c0 - b
                ok = all(0 <= oi + x < h and 0 <= oj + y < w and I[oi + x, oj + y] == vv
                         for vv, (x, y) in sub)
                if not ok:
                    continue
                key = frozenset((oi + x, oj + y) for _, (x, y) in var)
                if key in done:
                    continue
                done.add(key)
                bycol = {}
                for vv, (x, y) in fill:
                    rr, cc = oi + x, oj + y
                    if 0 <= rr < h and 0 <= cc < w and G[rr, cc] != vv:
                        bycol.setdefault(vv, []).append((rr, cc))
                for vv in sorted(bycol):
                    cells = sorted(bycol[vv])
                    ops.append(int(vv)); sels.append(sel_of(cells))
                    for rr, cc in cells:
                        G[rr, cc] = vv

    ops.append(34); sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
