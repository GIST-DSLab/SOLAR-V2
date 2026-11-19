"""
ARC Task: 6aa20dc0 (RE-ARC) — LLM-generated grid_maker
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
    bgc, fgc, c1, c2 = random.sample(cols, 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    od = unifint(diff_lb, diff_ub, (2, 4))
    od = min(od, min(h, w))
    ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    loci = randint(0, h - od)
    locj = randint(0, w - od)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = asindices(gi)
    inds = inds - backdrop(outbox(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ = 0
    tr = 0
    maxtr = 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        mf1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf = compose(mf2, mf1)
        cobj = normalize(upscale(mf(obj), fac))
        ohx, owx = shape(cobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - ohx and ij[1] <= w - owx)
        if len(cands) == 0:
            continue
        locc = choice(totuple(cands))
        cobjo = shift(cobj, locc)
        cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
        cobjoi = toindices(cobjo)
        if cobjoi.issubset(inds):
            succ += 1
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bg = Counter(I.ravel().tolist()).most_common(1)[0][0]

    # --- 1. locate the one COMPLETE symbol in I -------------------------------
    # It is an od x od block (od in 2..4) holding two distinct "corner" colours,
    # once each, at opposite corners of the block, every other non-background cell
    # inside it a single third colour (the fill colour), the block diagonally
    # symmetric, and the block surrounded by a ring of background.
    tmpl = None
    for od in (2, 3, 4):
        if od > h or od > w:
            continue
        for r in range(h - od + 1):
            for c in range(w - od + 1):
                W = I[r:r + od, c:c + od]
                ring_ok = True
                for rr in range(r - 1, r + od + 1):
                    for cc in range(c - 1, c + od + 1):
                        if r <= rr < r + od and c <= cc < c + od:
                            continue
                        if 0 <= rr < h and 0 <= cc < w and I[rr, cc] != bg:
                            ring_ok = False
                            break
                    if not ring_ok:
                        break
                if not ring_ok:
                    continue
                for a_pos, b_pos, sym in (((0, 0), (od - 1, od - 1), 'd'),
                                          ((0, od - 1), (od - 1, 0), 'c')):
                    a = W[a_pos]; b = W[b_pos]
                    if a == bg or b == bg or a == b:
                        continue
                    vals = Counter(W.ravel().tolist())
                    if vals[a] != 1 or vals[b] != 1:
                        continue
                    rest = [v for v in vals if v not in (a, b, bg)]
                    if len(rest) != 1 or vals[rest[0]] < 2:
                        continue
                    if sym == 'd':
                        if not np.array_equal(W, W.T):
                            continue
                    else:
                        if not np.array_equal(W, np.rot90(W.T, 2)):
                            continue
                    tmpl = (W.copy(), rest[0], (r, c))
                    break
                if tmpl is not None:
                    break
            if tmpl is not None:
                break
        if tmpl is not None:
            break

    if tmpl is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    T, fill, tpos = tmpl

    # --- 2. every scaled / mirrored version of that symbol --------------------
    variants = []
    seen = set()
    for fac in (1, 2, 3, 4):
        up = np.kron(T, np.ones((fac, fac), dtype=int))
        for V in (up, up.T, np.rot90(up.T, 2), np.flipud(up), np.fliplr(up)):
            key = V.tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(V)

    # --- 3. every incomplete copy: only its two corner marks are drawn in I,
    #        and every cell touching those marks is background -----------------
    tr0, tc0 = tpos
    tset = {(tr0 + i, tc0 + j) for i in range(T.shape[0]) for j in range(T.shape[1])}
    marks_needed = frozenset((r, c) for r in range(h) for c in range(w)
                             if I[r, c] != bg and (r, c) not in tset)
    forbidden = {(tr0 + i, tc0 + j)
                 for i in range(-1, T.shape[0] + 1) for j in range(-1, T.shape[1] + 1)}

    cands = []
    seen_place = set()
    for V in variants:
        m, n = V.shape
        marks = [(i, j) for i in range(m) for j in range(n)
                 if V[i, j] != bg and V[i, j] != fill]
        fills = [(i, j) for i in range(m) for j in range(n) if V[i, j] == fill]
        markset = set(marks)
        halo = set()
        for (i, j) in marks:
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (i + di, j + dj)
                if p not in markset:
                    halo.add(p)
        for r in range(h - m + 1):
            for c in range(w - n + 1):
                if any(I[r + i, c + j] != V[i, j] for (i, j) in marks):
                    continue
                bad = False
                for (i, j) in halo:
                    rr, cc = r + i, c + j
                    if 0 <= rr < h and 0 <= cc < w and I[rr, cc] != bg:
                        bad = True
                        break
                if bad:
                    continue
                mk = frozenset((r + i, c + j) for (i, j) in marks)
                if not mk <= marks_needed:
                    continue
                fl = tuple(sorted((r + i, c + j) for (i, j) in fills))
                cells = frozenset(fl) | mk
                if cells & forbidden:      # a real copy keeps clear of the model symbol
                    continue
                key = (mk, fl)
                if key in seen_place:
                    continue
                seen_place.add(key)
                cands.append({'r': r, 'c': c, 'm': m, 'n': n,
                              'marks': mk, 'fills': fl, 'cells': cells})

    # --- 4. the copies really present are the ones that use up every mark
    #        exactly once, without running into each other --------------------
    def compatible(a, b):
        if a['cells'] & b['cells']:
            return False
        ea = {(rr, cc) for rr in range(a['r'] - 1, a['r'] + a['m'] + 1)
              for cc in range(a['c'] - 1, a['c'] + a['n'] + 1)}
        eb = {(rr, cc) for rr in range(b['r'] - 1, b['r'] + b['m'] + 1)
              for cc in range(b['c'] - 1, b['c'] + b['n'] + 1)}
        return not (b['cells'] & ea) or not (a['cells'] & eb)

    by_mark = {}
    for cd in cands:
        for p in cd['marks']:
            by_mark.setdefault(p, []).append(cd)

    chosen = []
    budget = [200000]

    def cover(remaining):
        if not remaining:
            return True
        budget[0] -= 1
        if budget[0] < 0:
            return False
        pivot = min(remaining, key=lambda p: (len(by_mark.get(p, ())), p))
        for cd in sorted(by_mark.get(pivot, ()), key=lambda d: (d['r'], d['c'], d['m'])):
            if not cd['marks'] <= remaining:
                continue
            if any(not compatible(cd, o) for o in chosen):
                continue
            chosen.append(cd)
            if cover(remaining - cd['marks']):
                return True
            chosen.pop()
        return False

    if not cover(marks_needed):
        chosen = cands                      # no clean explanation: complete them all

    # --- 5. complete each copy: one Color(fill) per copy ----------------------
    chosen = sorted(chosen, key=lambda d: (d['r'], d['c']))
    cur = I.copy()
    for cd in chosen:
        cells = cd['fills']
        if not cells or all(cur[rr, cc] == fill for (rr, cc) in cells):
            continue
        ops.append(int(fill)); sels.append(sel_of(cells))
        for (rr, cc) in cells:
            cur[rr, cc] = fill

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
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
                                f"for task 6aa20dc0"
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
                    f"Failed to build a complete episode for task 6aa20dc0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6aa20dc0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
