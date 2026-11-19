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
    # The only colour role the rule depends on is the "noise" colour: the colour of the
    # rectangular patches that damage the periodic wallpaper and that the rule erases.
    # Fixing it for the whole episode makes it readable from the demonstrations
    # (it is the colour present in every example input and absent from every output).
    import random
    noisec = random.choice(list(range(10)))
    return {"noisec": noisec}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, noisec=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if noisec is None:
        noisec = choice(cols)

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))
    h = max(h, 6)
    w = max(w, 6)

    hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
    wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
    pinds = asindices(canvas(-1, (hp, wp)))

    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
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

    numpatches = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 20 * numpatches + 50
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
               len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2

    rotopts = [identity, rot180]
    if h <= max_w and w <= max_h:
        rotopts = rotopts + [rot90, rot270]
    rotf = choice(tuple(rotopts))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: the grid is a wallpaper pattern, periodic under a translation lattice, damaged by
    solid rectangular patches of one 'noise' colour.  Restore every damaged cell from the
    intact copy of the same pattern cell found by a lattice translation.

    Everything below is measured from I plus the demonstrations:
      * the noise colour  -> the colour every example input has and every example output lacks
      * the lattice       -> translations under which I's undamaged cells agree with themselves
    O is never inspected.
    """
    import numpy as np
    from collections import deque
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    A = np.asarray(I, dtype=int)
    h, w = A.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]          # whole-grid rectangle (bbox is exact here)

    # ---------- helpers -------------------------------------------------------
    def ncomp(col):
        m = (A == col)
        seen = np.zeros((h, w), dtype=bool)
        n = 0
        for r in range(h):
            for c in range(w):
                if m[r, c] and not seen[r, c]:
                    n += 1
                    dq = deque([(r, c)])
                    seen[r, c] = True
                    while dq:
                        cr, cc = dq.popleft()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                dq.append((nr, nc))
        return n

    def blobbiness(col):
        # noise = solid patches -> very few connected components, pattern colours -> many
        return (ncomp(col), int((A == col).sum()), int(col))

    # ---------- 1. the noise colour, read from the demonstrations -------------
    noisec = None
    votes = {}
    if examples:
        for pair in examples:
            try:
                ei = np.asarray(pair[0], dtype=int)
                eo = np.asarray(pair[1], dtype=int)
            except Exception:
                continue
            if ei.shape == A.shape and np.array_equal(ei, A):
                continue                      # never learn from the pair being derived
            gone = set(np.unique(ei).tolist()) - set(np.unique(eo).tolist())
            for c in gone:
                votes[c] = votes.get(c, 0) + 1
    present = [c for c in votes if bool((A == c).any())]
    if present:
        best = max(votes[c] for c in present)
        top = [c for c in present if votes[c] == best]
        noisec = top[0] if len(top) == 1 else min(top, key=blobbiness)
    if noisec is None:                        # no demos: fall back to the structural signature
        noisec = min(set(np.unique(A).tolist()), key=blobbiness)

    ok = (A != noisec)
    if bool(ok.all()):                        # nothing damaged
        ops.append(34); sels.append(full)
        return ops, sels

    # ---------- 2. the pattern's translation lattice, measured on I -----------
    def agrees(dr, dc):
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r1 <= r0 or c1 <= c0:
            return False, 0
        a = A[r0:r1, c0:c1]
        b = A[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m = ok[r0:r1, c0:c1] & ok[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        n = int(m.sum())
        if n == 0:
            return False, 0
        return bool(np.array_equal(a[m], b[m])), n

    need = max(12, int(0.25 * int(ok.sum())))
    cand = []
    for dr in range(0, h // 2 + 1):
        for dc in range(-(w // 2), w // 2 + 1):
            if dr == 0 and dc <= 0:
                continue
            good, n = agrees(dr, dc)
            if good and n >= need:
                cand.append((dr * dr + dc * dc, dr, dc))
    cand.sort()

    vecs = []
    if cand:
        v1 = (cand[0][1], cand[0][2])
        v2 = None
        for _, dr, dc in cand[1:]:
            if dr * v1[1] - dc * v1[0] != 0:      # independent of v1
                v2 = (dr, dc)
                break
        lim = h + w
        combos = set()
        brange = [0] if v2 is None else range(-lim, lim + 1)
        for a in range(-lim, lim + 1):
            ar, ac = a * v1[0], a * v1[1]
            for b in brange:
                tr = ar + (b * v2[0] if v2 is not None else 0)
                tc = ac + (b * v2[1] if v2 is not None else 0)
                if (tr or tc) and abs(tr) < h and abs(tc) < w:
                    combos.add((tr, tc))
        vecs = sorted(combos, key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1])))

    # ---------- 3. what each damaged cell must become ------------------------
    damaged = [(r, c) for r in range(h) for c in range(w) if not ok[r, c]]
    pred = {}
    for (r, c) in damaged:
        for (tr, tc) in vecs:
            rr, cc = r + tr, c + tc
            if 0 <= rr < h and 0 <= cc < w and ok[rr, cc]:
                pred[(r, c)] = int(A[rr, cc])
                break

    # ---------- 4. repair one damaged patch at a time ------------------------
    seen = set()
    comps = []
    for cell in damaged:
        if cell in seen:
            continue
        seen.add(cell)
        dq = deque([cell])
        comp = []
        while dq:
            cr, cc = dq.popleft()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not ok[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dq.append((nr, nc))
        comps.append(sorted(comp))

    groups = []                               # (colour, cells) in patch-by-patch order
    for comp in comps:
        by = {}
        for cell in comp:
            if cell in pred:
                by.setdefault(pred[cell], []).append(cell)
        for col in sorted(by):
            groups.append((col, by[col]))

    if len(groups) > 120:                     # very many patches: one pass per colour
        merged = {}
        for col, cells in groups:
            merged.setdefault(col, []).extend(cells)
        groups = [(col, sorted(merged[col])) for col in sorted(merged)]

    for col, cells in groups:
        ops.append(int(col))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append(full)
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
