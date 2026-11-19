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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# helpers (plain-python versions of the RE-ARC DSL pieces the generator uses)
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    rng = b - a
    lo = a + int(round(diff_lb * rng))
    hi = a + int(round(diff_ub * rng))
    lo = max(a, min(b, lo))
    hi = max(lo, min(b, hi))
    return random.randint(lo, hi)


def _dmirror(obj):
    return {(c, r): v for (r, c), v in obj.items()}


def _hmirror(obj, H):
    return {(H - 1 - r, c): v for (r, c), v in obj.items()}


def _vmirror(obj, W):
    return {(r, W - 1 - c): v for (r, c), v in obj.items()}


def _cmirror(obj, H, W):
    return {(W - 1 - c, H - 1 - r): v for (r, c), v in obj.items()}


def _apply_mf(obj, name, H, W):
    if name == 'identity':
        return dict(obj), H, W
    if name == 'dmirror':
        return _dmirror(obj), W, H
    if name == 'cmirror':
        return _cmirror(obj, H, W), W, H
    if name == 'hmirror':
        return _hmirror(obj, H), H, W
    return _vmirror(obj, W), H, W


def _upscale(obj, f):
    out = {}
    for (r, c), v in obj.items():
        for dr in range(f):
            for dc in range(f):
                out[(r * f + dr, c * f + dc)] = v
    return out


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, c1, c2 = random.sample(range(10), 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, c1=None, c2=None, **kw) -> dict:
    if None in (bgc, fgc, c1, c2):
        bgc, fgc, c1, c2 = random.sample(range(10), 4)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    od = _unifint(diff_lb, diff_ub, (2, 4))
    ncellsextra = random.randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = {(i, j) for i in range(od) for j in range(od)}
    pool = sorted(sinds - {(0, 0), (od - 1, od - 1)})
    extracells = set(random.sample(pool, min(ncellsextra, len(pool))))

    def dn(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    extracells.add(random.choice(sorted(dn((0, 0)) & sinds)))
    extracells.add(random.choice(sorted(dn((od - 1, od - 1)) & sinds)))
    obj = {(0, 0): c1, (od - 1, od - 1): c2}
    for p in extracells:
        obj[p] = fgc
    obj.update(_dmirror(obj))
    if random.choice((True, False)):
        obj = _hmirror(obj, od)

    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - od)
    locj = random.randint(0, w - od)
    for (r, c), v in obj.items():
        gi[loci + r][locj + c] = v
    go = [row[:] for row in gi]

    inds = {(i, j) for i in range(h) for j in range(w)}
    inds -= {(i, j) for i in range(loci - 1, loci + od + 1) for j in range(locj - 1, locj + od + 1)}

    nocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ, tr, maxtr = 0, 0, 4 * nocc
    names = ('identity', 'dmirror', 'vmirror', 'cmirror', 'hmirror')
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = random.randint(1, 4)
        m1 = random.choice(names)
        m2 = random.choice(names)
        o1, H1, W1 = _apply_mf(obj, m1, od, od)
        o2, H2, W2 = _apply_mf(o1, m2, H1, W1)
        cobj = _upscale(o2, fac)
        ohx, owx = H2 * fac, W2 * fac
        cands = sorted(p for p in inds if p[0] <= h - ohx and p[1] <= w - owx)
        if not cands:
            continue
        li, lj = random.choice(cands)
        cells = {(li + r, lj + c): v for (r, c), v in cobj.items()}
        if set(cells).issubset(inds):
            succ += 1
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            inds -= {(i, j) for i in range(min(rs) - 1, max(rs) + 2)
                     for j in range(min(cs) - 1, max(cs) + 2)}
            for (r, c), v in cells.items():
                go[r][c] = v
                if v != fgc:
                    gi[r][c] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # The golf entry and the maker read this task the same way: the template is
    # stamped (scaled 1..4, in the 5 mirror orientations) wherever its corner
    # markers appear with an empty body. The golf entry rewrites the markers too,
    # but they are already there, so that changes nothing. In both readings, what
    # gets painted is the template body at each matched site. So there is
    # nothing to change: one fgc Color op per site, covering that site's body
    # cells, which come from the template's shape and the marker positions in I.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- 1) template = multicolour 8-connected component(s) with max #colours
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack, comp = [(r, c)], []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)

    if not comps:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    ncol = [len({int(I[r, c]) for r, c in comp}) for comp in comps]
    mx = max(ncol)
    tmpl_cells = [p for comp, n in zip(comps, ncol) if n == mx for p in comp]
    tr0 = min(p[0] for p in tmpl_cells); tr1 = max(p[0] for p in tmpl_cells)
    tc0 = min(p[1] for p in tmpl_cells); tc1 = max(p[1] for p in tmpl_cells)

    base = {}
    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            if I[r, c] != bgc:
                base[(r - tr0, c - tc0)] = int(I[r, c])
    if not base:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    BH, BW = tr1 - tr0 + 1, tc1 - tc0 + 1

    fgc = Counter(base.values()).most_common(1)[0][0]

    # ---- 2) 20 legal stamps: upscale 1..4 x {id, dmirror, cmirror, hmirror, vmirror}
    def mirror(cells, H, W, mf):
        if mf == 'identity':
            return {(r, c): v for (r, c), v in cells.items()}, H, W
        if mf == 'dmirror':
            return {(c, r): v for (r, c), v in cells.items()}, W, H
        if mf == 'cmirror':
            return {(W - 1 - c, H - 1 - r): v for (r, c), v in cells.items()}, W, H
        if mf == 'hmirror':
            return {(H - 1 - r, c): v for (r, c), v in cells.items()}, H, W
        return {(r, W - 1 - c): v for (r, c), v in cells.items()}, H, W

    units = []
    for f in range(1, 5):
        up = {}
        for (r, c), v in base.items():
            for dr in range(f):
                for dc in range(f):
                    up[(r * f + dr, c * f + dc)] = v
        UH, UW = BH * f, BW * f
        for mf in ('identity', 'dmirror', 'cmirror', 'hmirror', 'vmirror'):
            units.append(mirror(up, UH, UW, mf))

    # ---- 3) occurrences: markers present in I, body still empty, isolated ring
    sites = []
    seen_sets = set()
    for cells, UH, UW in units:
        marks = {p: v for p, v in cells.items() if v != fgc}
        body = [p for p, v in cells.items() if v == fgc]
        if not marks or not body:
            continue
        mset = set(marks)
        ring = set()
        for (r, c) in mset:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r + dr, c + dc) not in mset:
                    ring.add((r + dr, c + dc))
        for r0 in range(h - UH + 1):
            for c0 in range(w - UW + 1):
                if any(I[r0 + r, c0 + c] != v for (r, c), v in marks.items()):
                    continue
                if any(I[r0 + r, c0 + c] != bgc for (r, c) in body):
                    continue
                bad = False
                for (r, c) in ring:
                    rr, cc = r0 + r, c0 + c
                    if 0 <= rr < h and 0 <= cc < w and I[rr, cc] != bgc:
                        bad = True
                        break
                if bad:
                    continue
                filled = frozenset((r0 + r, c0 + c) for (r, c) in body)
                if filled in seen_sets:
                    continue
                seen_sets.add(filled)
                sites.append((r0, c0, sorted(filled)))

    # ---- 4) stamp the template body at each matched site, one op per site
    sites.sort(key=lambda s: (s[0], s[1]))
    for _r0, _c0, filled in sites:
        ops.append(int(fgc))
        sels.append(sel_of(filled))

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
