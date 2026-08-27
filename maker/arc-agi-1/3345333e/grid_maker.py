"""
ARC Task: 3345333e (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of

VARIANTS = [{"orient": "v"}, {"orient": "h"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, occcol = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "occcol": occcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, occcol, orient=None) -> dict:
    if orient is None:
        orient = random.choice(("v", "h"))

    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def hw(s):
        rs = [i for i, _ in s]
        cs = [j for _, j in s]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def nbrs(s):
        out = set()
        for (i, j) in s:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out

    cap = min(max_h, max_w)                      # transposing mirrors may swap h and w
    h = unifint(diff_lb, diff_ub, (10, max(10, cap)))
    w = unifint(diff_lb, diff_ub, (10, max(10, cap)))
    oh = unifint(diff_lb, diff_ub, (4, h - 2))
    ow = unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))

    shp = {(0, 0)}
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    for _ in range(nc):
        cand = sorted((bounds - shp) & nbrs(shp))
        if not cand:
            break
        shp.add(random.choice(cand))
    while hw(shp)[0] < 3 or hw(shp)[1] < 3:
        cand = sorted((bounds - shp) & nbrs(shp))
        if not cand:
            break
        shp.add(random.choice(cand))

    cjs = [j for _, j in shp]
    cmin, cmax = min(cjs), max(cjs)
    wshp = cmax - cmin + 1
    vmshp = {(i, cmin + cmax - j) for i, j in shp}
    if random.choice((True, False)):
        vmshp = {(i, j) for (i, j) in vmshp if j != wshp - 1}
    wvm = hw(vmshp)[1]
    comb = shp | {(i, j - wvm) for i, j in vmshp}
    rmn = min(i for i, _ in comb)
    cmn = min(j for _, j in comb)
    shp = {(i - rmn, j - cmn) for i, j in comb}
    oh, ow = hw(shp)

    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    shp = {(i + loci, j + locj) for i, j in shp}

    go = [[bgc] * w for _ in range(h)]
    for (i, j) in shp:
        go[i][j] = objc

    boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(diff_lb, diff_ub, (2, ow // 2))
    ulci = random.randint(loci - 1, loci + oh - boxh + 1)
    ulcj = random.randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    gi = [row[:] for row in go]
    for i in range(ulci, ulci + boxh):
        for j in range(ulcj, ulcj + boxw):
            gi[i][j] = occcol

    def xf(g, name):
        if name == "identity":
            return [row[:] for row in g]
        if name == "vmirror":
            return [row[::-1] for row in g]
        if name == "hmirror":
            return [row[:] for row in g[::-1]]
        if name == "rot180":
            return [row[::-1] for row in g[::-1]]
        if name == "dmirror":
            return [list(r) for r in zip(*g)]
        if name == "cmirror":
            return [list(r) for r in zip(*[row[::-1] for row in g[::-1]])]
        if name == "rot90":
            return [list(r) for r in zip(*g[::-1])]
        if name == "rot270":
            return [list(r) for r in zip(*g)][::-1]
        return g

    keep = ["identity", "vmirror", "hmirror", "rot180"]      # keeps the mirror axis vertical
    swap = ["dmirror", "cmirror", "rot90", "rot270"]         # turns the mirror axis horizontal
    if orient == "v":
        fns = random.sample(keep, random.choice((1, 2)))
    else:
        fns = [random.choice(swap)]
        if random.choice((True, False)):
            fns.append(random.choice(keep))
    for fn in fns:
        gi = xf(gi, fn)
        go = xf(go, fn)

    return {"input": [[int(v) for v in row] for row in gi],
            "output": [[int(v) for v in row] for row in go]}


def derive_operations(I, O):
    """A solid rectangle of one colour covers part of a mirror-symmetric figure.
    Wipe the rectangle away and redraw the piece of the figure it was hiding,
    found by reflecting the still-visible part onto itself."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background = the colour the canvas was painted with; it rings the whole grid
    border = ([int(I[0, j]) for j in range(w)] + [int(I[h - 1, j]) for j in range(w)] +
              [int(I[i, 0]) for i in range(1, h - 1)] + [int(I[i, w - 1]) for i in range(1, h - 1)])
    bgc = Counter(border).most_common(1)[0][0]
    others = sorted(set(int(v) for v in I.flatten().tolist()) - {bgc})
    bg_arr = (I == bgc)

    def cells_of(col):
        rs, cs = np.nonzero(I == col)
        return set(zip(rs.tolist(), cs.tolist()))

    def is_rect(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    # The occluder is the colour whose cells fill a solid rectangle; the remaining
    # non-background colour is the figure. (Both colours qualify only when the figure
    # itself happens to be a solid block — an ambiguity the task's own reference
    # implementation resolves arbitrarily, so O settles it below.)
    best = None
    for occ in others:
        rest = [c for c in others if c != occ]
        if len(rest) != 1:
            continue
        obj = rest[0]
        box = cells_of(occ)
        objcells = cells_of(obj)
        if not box or not objcells or not is_rect(box):
            continue
        obj_arr = (I == obj)

        rr = np.array([r for r, _ in sorted(objcells)])
        cc = np.array([c for _, c in sorted(objcells)])
        rmin, rmax, cmin, cmax = rr.min(), rr.max(), cc.min(), cc.max()
        mirrors = [(rr, cmin + cmax - cc),          # reflected left<->right
                   (rmin + rmax - rr, cc)]          # reflected up<->down

        union = objcells | box
        ur = [r for r, _ in union]
        uc = [c for _, c in union]
        lim = max((max(ur) - min(ur) + 1) // 2 + 1, (max(uc) - min(uc) + 1) // 2 + 1)

        # Slide each reflection over the figure: it may never land on background,
        # and the right placement is the one covering the most of the visible figure.
        cands = []
        for (mr, mc) in mirrors:
            for di in range(-lim, lim + 1):
                for dj in range(-lim, lim + 1):
                    ar, ac = mr + di, mc + dj
                    inside = (ar >= 0) & (ar < h) & (ac >= 0) & (ac < w)
                    ir, ic = ar[inside], ac[inside]
                    if bg_arr[ir, ic].any():
                        continue
                    cands.append((int(obj_arr[ir, ic].sum()),
                                  set(zip(ir.tolist(), ic.tolist()))))
        if not cands:
            continue
        cands.sort(key=lambda t: -t[0])
        for score, sh in cands:
            paint = sh & box            # outside the rectangle the reflection is already drawn
            G = I.copy()
            for (r, c) in box:
                G[r, c] = bgc
            for (r, c) in paint:
                G[r, c] = obj
            agrees = bool(np.array_equal(G, O))
            cand = (agrees, score, obj, box, paint)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
            if agrees:                  # best overlap wins; O only settles exact ties
                break
        if best is not None and best[0]:
            break

    if best is None:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    _, _, obj, box, paint = best

    # 1) wipe the occluding rectangle back to background
    ops.append(int(bgc))
    sels.append(sel_of(sorted(box)))

    # 2) redraw the hidden part of the figure, one connected patch at a time
    todo = set(paint)
    comps = []
    while todo:
        seed = min(todo)
        todo.discard(seed)
        comp = [seed]
        q = deque([seed])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in todo:
                        todo.discard(n)
                        comp.append(n)
                        q.append(n)
        comps.append(sorted(comp))
    comps.sort(key=lambda cp: cp[0])
    for comp in comps:
        ops.append(int(obj))
        sels.append(sel_of(comp))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3345333e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3345333e"
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
                                f"for task 3345333e"
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
                    f"Failed to build a complete episode for task 3345333e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3345333e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
