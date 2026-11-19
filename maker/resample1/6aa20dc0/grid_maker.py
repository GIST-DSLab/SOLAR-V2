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
try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        return {"cells": [list(map(int, c)) for c in cells]}


def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    v = a + int(round(d * (b - a)))
    return max(a, min(b, v))


def _dihedral(cells, n, k):
    # cells: dict (r,c)->col in an n x n square; k in 0..7 (all 8 symmetries)
    out = {}
    for (r, c), v in cells.items():
        rr, cc = r, c
        if k & 4:
            rr, cc = cc, rr
        if k & 1:
            rr = n - 1 - rr
        if k & 2:
            cc = n - 1 - cc
        out[(rr, cc)] = v
    return out


def _upscale(cells, fac):
    out = {}
    for (r, c), v in cells.items():
        for a in range(fac):
            for b in range(fac):
                out[(r * fac + a, c * fac + b)] = v
    return out


def _find(I, bgc, fgc):
    """Locate the template (fgc cells + 2 corner cells) and every partial copy
    (only corner colours present, scaled/mirrored). Returns list of fill-cell tuples."""
    hi, wi = I.shape
    F = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc]
    if not F:
        return None
    corners = set()
    for (r, c) in F:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y, x = r + dr, c + dc
            if 0 <= y < hi and 0 <= x < wi and I[y, x] != bgc and I[y, x] != fgc:
                corners.add((y, x))
    if len(corners) != 2:
        return None
    allc = set(F) | corners
    r0 = min(r for r, _ in allc); c0 = min(c for _, c in allc)
    r1 = max(r for r, _ in allc); c1 = max(c for _, c in allc)
    if r1 - r0 != c1 - c0 or not (1 <= r1 - r0 <= 3):
        return None
    od = r1 - r0 + 1
    tmpl = {(r - r0, c - c0): int(I[r, c]) for (r, c) in allc}
    cp = sorted((r - r0, c - c0) for r, c in corners)
    if cp not in ([(0, 0), (od - 1, od - 1)], [(0, od - 1), (od - 1, 0)]):
        return None
    found, done = [], set()
    for k in range(8):
        tk = _dihedral(tmpl, od, k)
        for fac in range(1, 5):
            T = _upscale(tk, fac)
            S = od * fac
            if S > hi or S > wi:
                continue
            P = {p: v for p, v in T.items() if v != fgc}
            fill = [p for p, v in T.items() if v == fgc]
            exp = np.full((S, S), bgc, dtype=int)
            for (a, b), v in P.items():
                exp[a, b] = v
            (pr, pc), pv = next(iter(P.items()))
            for i in range(hi - S + 1):
                for j in range(wi - S + 1):
                    if I[i + pr, j + pc] != pv:
                        continue
                    if not (I[i:i + S, j:j + S] == exp).all():
                        continue
                    cells = tuple(sorted((i + a, j + b) for a, b in fill))
                    if cells in done:
                        continue
                    done.add(cells)
                    found.append(cells)
    return found


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, c1, c2 = random.sample(range(10), 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    od = _unifint(diff_lb, diff_ub, (2, 4))
    ncellsextra = random.randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = {(i, j) for i in range(od) for j in range(od)}
    pool = sorted(sinds - {(0, 0), (od - 1, od - 1)})
    extracells = set(random.sample(pool, min(ncellsextra, len(pool))))
    extracells.add(random.choice([(0, 1), (1, 0)]))
    extracells.add(random.choice([(od - 2, od - 1), (od - 1, od - 2)]))
    obj = {e: fgc for e in extracells}
    obj[(0, 0)] = c1
    obj[(od - 1, od - 1)] = c2
    obj.update({(c, r): v for (r, c), v in list(obj.items())})
    if random.choice((True, False)):
        obj = {(od - 1 - r, c): v for (r, c), v in obj.items()}
    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - od)
    locj = random.randint(0, w - od)
    for (r, c), v in obj.items():
        gi[r + loci, c + locj] = v
    go = gi.copy()
    inds = {(i, j) for i in range(h) for j in range(w)}
    inds -= {(i, j) for i in range(loci - 1, loci + od + 1) for j in range(locj - 1, locj + od + 1)}
    nocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ, tr, maxtr = 0, 0, 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = random.randint(1, 4)
        cobj = _upscale(_dihedral(obj, od, random.randint(0, 7)), fac)
        S = od * fac
        cands = [ij for ij in inds if ij[0] <= h - S and ij[1] <= w - S]
        if not cands:
            continue
        a, b = random.choice(cands)
        box = {(a + i, b + j) for i in range(S) for j in range(S)}
        if box.issubset(inds):
            succ += 1
            inds -= {(i, j) for i in range(a - 1, a + S + 1) for j in range(b - 1, b + S + 1)}
            for (r, c), v in cobj.items():
                go[a + r, b + c] = v
                if v != fgc:
                    gi[a + r, b + c] = v
    if succ == 0:
        raise ValueError("no occurrence placed")
    # reject ambiguous layouts (pieces of different copies forming a fake copy)
    found = _find(gi, bgc, fgc) or []
    pred = gi.copy()
    for cells in found:
        for r, c in cells:
            pred[r, c] = fgc
    if not (pred == go).all() or sum(len(x) for x in found) != int((go != gi).sum()):
        raise ValueError("ambiguous layout")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # fill colour: read from the demonstrations (the colour the rule adds)
    fgc = None
    if examples:
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                cnt.update(eo[ei != eo].tolist())
        if cnt:
            fgc = cnt.most_common(1)[0][0]
    if fgc is None or not (I == fgc).any():
        # fallback: the fill colour is the one that forms a valid template in I
        best = None
        for f in sorted(set(I.flatten().tolist()) - {bgc}):
            res = _find(I, bgc, f)
            if res is not None and (best is None or len(res) > best[0]):
                best = (len(res), f)
        fgc = best[1] if best else None
    occ = (_find(I, bgc, fgc) if fgc is not None else None) or []
    ops, sels = [], []
    # one Color(fgc) per detected partial copy: paint that copy's template fill cells
    for cells in occ:
        ops.append(int(fgc)); sels.append(sel_of(list(cells)))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
