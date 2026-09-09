"""
ARC Task: 97a05b5b (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback for standalone use
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    """bgc = canvas background, sqc = colour of the big rectangle (and of the
    'holes' drawn inside the small key patterns).  Both roles are structural, so
    they must stay fixed across the whole episode.  The per-object colours are
    arbitrary and carried by the rule itself, so they are re-sampled per instance
    (the full 0..9 palette stays available -- 0 may legally be any role)."""
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


# ----------------------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int = 30, max_w: int = 30,
             bgc: int = 0, sqc: int = 1) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(15, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(15, max_w), max_w))
    sgh = randint(h // 3, h // 3 * 2)
    sgw = randint(w // 3, w // 3 * 2)
    sgh = max(4, min(sgh, h))
    sgw = max(4, min(sgw, w))
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = randint(2, sgh // 2)
    ow = randint(2, sgw // 2)
    nobjs = unifint(diff_lb, diff_ub, (1, 8))
    objs = set()
    cands = asindices(canvas(-1, (oh, ow)))
    forbidden = set()
    tr = 0
    maxtr = 4 * nobjs
    while len(objs) != nobjs and tr < maxtr:
        tr += 1
        obj = {choice(totuple(cands))}
        ncells = randint(1, oh * ow - 1)
        for k in range(ncells - 1):
            rem = (cands - obj) & mapply(neighbors, obj)
            if len(rem) == 0:
                break
            obj.add(choice(totuple(rem)))
        obj |= choice((dmirror, cmirror, vmirror, hmirror))(obj)
        if len(obj) == height(obj) * width(obj):
            continue
        obj = frozenset(obj)
        objn = normalize(obj)
        if objn not in forbidden:
            objs.add(objn)
        for augmf1 in (identity, dmirror, cmirror, hmirror, vmirror):
            for augmf2 in (identity, dmirror, cmirror, hmirror, vmirror):
                forbidden.add(augmf1(augmf2(objn)))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    bd = backdrop(frozenset({(loci, locj), (loci + sgh - 1, locj + sgw - 1)}))
    gi = fill(gi, sqc, bd)
    go = canvas(sqc, (sgh, sgw))
    goinds = asindices(go)
    giinds = asindices(gi) - shift(goinds, (loci, locj))
    giinds = giinds - mapply(neighbors, shift(goinds, (loci, locj)))
    while succ < nobjs and tr < maxtr and len(objs) > 0:
        tr += 1
        obj = choice(totuple(objs))
        col = choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        f1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f = compose(f1, f2)
        subgo = f(subgi)
        giobj = asobject(subgi)
        goobj = asobject(subgo)
        ohi, owi = shape(giobj)
        oho, owo = shape(goobj)
        gocands = sfilter(goinds, lambda ij: ij[0] <= sgh - oho and ij[1] <= sgw - owo)
        if len(gocands) == 0:
            continue
        goloc = choice(totuple(gocands))
        goplcd = shift(goobj, goloc)
        goplcdi = toindices(goplcd)
        if goplcdi.issubset(goinds):
            gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= owi)
            if len(gicands) == 0:
                continue
            giloc = choice(totuple(gicands))
            giplcd = shift(giobj, giloc)
            giplcdi = toindices(giplcd)
            if giplcdi.issubset(giinds):
                succ += 1
                remcols = remove(col, remcols)
                objs = remove(obj, objs)
                goinds = goinds - goplcdi
                giinds = (giinds - giplcdi) - mapply(neighbors, giplcdi)
                gi = paint(gi, giplcd)
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)), lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------- derivation
def _components(I, bgc):
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for a, b in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if 0 <= a < hi and 0 <= b < wi and not seen[a, b] and I[a, b] != bgc:
                            seen[a, b] = True
                            stack.append((a, b))
                comps.append(cells)
    return comps


def _variants(g):
    out, seen = [], set()
    for k in range(4):
        for fl in (False, True):
            t = np.rot90(g, k)
            if fl:
                t = np.fliplr(t)
            t = np.ascontiguousarray(t)
            key = (t.shape, t.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return out


def _analyze(I, bgc, strict=True):
    """Split I into the big one-colour rectangle (with bgc holes punched into it)
    and the small full-rectangle two-colour 'key' patterns lying outside it, then
    locate, for every key pattern, the hole it matches (under the 8 dihedral
    transforms).  Everything is read from I only."""
    hi, wi = I.shape
    keys, rest = [], []
    for cells in _components(I, bgc):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        pal = set(int(I[r, c]) for r, c in cells)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(pal) == 2:
            keys.append((r0, c0, r1, c1))
        else:
            rest.extend(cells)
    if not rest:
        return None
    pal = set(int(I[r, c]) for r, c in rest)
    if len(pal) != 1:
        return None
    sqc = pal.pop()
    rs = [p[0] for p in rest]
    cs = [p[1] for p in rest]
    R0, R1, C0, C1 = min(rs), max(rs), min(cs), max(cs)
    reg = I[R0:R1 + 1, C0:C1 + 1]
    if not set(int(v) for v in np.unique(reg)) <= {sqc, bgc}:
        return None
    rh, rw = reg.shape
    holes = {(r, c) for r in range(rh) for c in range(rw) if reg[r, c] == bgc}

    cand_lists = []
    for (r0, c0, r1, c1) in keys:
        g = I[r0:r1 + 1, c0:c1 + 1]
        gp = set(int(v) for v in np.unique(g))
        if sqc not in gp or len(gp) != 2:
            return None
        col = [x for x in gp if x != sqc][0]
        cands = []
        for t in _variants(g):
            th, tw = t.shape
            mask = (t == sqc)
            for rr in range(rh - th + 1):
                for cc in range(rw - tw + 1):
                    sub = reg[rr:rr + th, cc:cc + tw]
                    if np.array_equal(sub == bgc, mask):
                        cands.append((rr, cc, t, col))
        if not cands:
            return None
        cand_lists.append(cands)

    n = len(cand_lists)
    result = [None] * n
    order = sorted(range(n), key=lambda i: len(cand_lists[i]))

    def rec(k, remaining):
        if k == n:
            return (not strict) or (len(remaining) == 0)
        i = order[k]
        for (rr, cc, t, col) in cand_lists[i]:
            hs = {(rr + a, cc + b) for a in range(t.shape[0]) for b in range(t.shape[1])
                  if t[a, b] == sqc}
            if hs <= remaining:
                result[i] = (rr, cc, t, col)
                if rec(k + 1, remaining - hs):
                    return True
        return False

    if not rec(0, set(holes)):
        return None
    return (R0, C0, R1, C1, sqc, [p for p in result if p is not None])


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    freq = [c for c, _ in Counter(I.flatten().tolist()).most_common()]
    info = None
    for strict in (True, False):
        for cand_bg in freq:
            info = _analyze(I, int(cand_bg), strict=strict)
            if info is not None:
                bgc = int(cand_bg)
                break
        if info is not None:
            break

    ops, sels = [], []

    if info is None:
        # extreme fallback: nothing recognisable -- just hand back the grid
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    R0, C0, R1, C1, sqc, placements = info
    rh, rw = R1 - R0 + 1, C1 - C0 + 1

    # one key pattern at a time: erase its hole (paint it the rectangle's colour),
    # then draw the pattern's own colour on the cells around that hole.
    for (rr, cc, t, col) in sorted(placements, key=lambda p: (p[0], p[1])):
        th, tw = t.shape
        hole_cells, body_cells = [], []
        for a in range(th):
            for b in range(tw):
                cell = (R0 + rr + a, C0 + cc + b)
                if t[a, b] == sqc:
                    hole_cells.append(cell)
                else:
                    body_cells.append(cell)
        if hole_cells:
            ops.append(int(sqc)); sels.append(sel_of(hole_cells))
        if body_cells:
            ops.append(int(col)); sels.append(sel_of(body_cells))

    # keep only the big rectangle; selection is exactly that full rectangle
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])
    ops.append(34); sels.append([0, 0, rh - 1, rw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
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
                                f"for task 97a05b5b"
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
                    f"Failed to build a complete episode for task 97a05b5b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"97a05b5b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
