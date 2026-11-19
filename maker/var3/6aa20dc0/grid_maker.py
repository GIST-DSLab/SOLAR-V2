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
    hlo, wlo = min(10, max_h), min(10, max_w)

    def readable(gi_t, go_t):
        """True when the markers standing on gi_t admit exactly the completion
        go_t and no other -- run the very search a solver runs and see whether
        it lands on go_t.  A copy that lets one copy's c1 marker pair up with
        another copy's c2 marker would make the puzzle two-answered, so the
        placement that created the pairing is dropped instead of shipped."""
        A = np.asarray(gi_t, dtype=int)
        B = np.asarray(go_t, dtype=int)
        ops, sels = derive_operations(A, B)
        g = A.copy()
        for op, sel in zip(ops, sels):
            if op == 34:
                continue
            for rr, cc in sel["cells"]:
                g[rr, cc] = op
        return np.array_equal(g, B)

    gi = go = None
    for _attempt in range(40):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        od = unifint(diff_lb, diff_ub, (2, min(4, h, w)))
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
        maxtr = 4 * nocc + 8
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
            if not cobjoi.issubset(inds):
                continue
            gi2 = paint(gi, cobji)
            go2 = paint(go, cobjo)
            if not readable(gi2, go2):
                continue
            gi, go = gi2, go2
            inds = inds - backdrop(outbox(cobjoi))
            succ += 1
        if succ >= 1:
            return {'input': gi, 'output': go}
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Finish every half-drawn copy of the key.

    The rule is fixed: find the one complete key, then wherever only its two
    corner markers were left standing (at any of the 4 scales and 5 mirrorings)
    paint the key's body back in.  What is NOT fixed is the route: the copies
    are independent of one another, so which one a solver starts with, which
    way the sweep runs, and whether a copy is filled in one stroke or arm by
    arm are free choices.  They are taken here from the input itself -- its
    size, its three colours, where the key sits, how many copies there are --
    so a different puzzle is walked a different way and the same puzzle is
    always walked the same way.  (There is no base-then-draw-over choice to
    make: nothing in this task is ever cleared, every cell that changes goes
    from background straight to the body colour, once.)
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ---- background: canvas colour the generator paints before placing anything
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- 1) find the KEY: multicolour diagonal components, max #colours
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

    # body colour of the key = its most common non-background colour
    fgc = Counter(base.values()).most_common(1)[0][0]

    # ---- 2) build the 20 legal stamps: upscale 1..4 x {id,dmirror,cmirror,hmirror,vmirror}
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
            cells, H2, W2 = mirror(up, UH, UW, mf)
            units.append((cells, H2, W2))

    # ---- 3) locate occurrences: the marker cells present in I, body cells still empty
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
                ok = True
                for (r, c), v in marks.items():
                    if I[r0 + r, c0 + c] != v:
                        ok = False
                        break
                if not ok:
                    continue
                for (r, c) in body:
                    if I[r0 + r, c0 + c] != bgc:
                        ok = False
                        break
                if not ok:
                    continue
                for (r, c) in ring:
                    rr, cc = r0 + r, c0 + c
                    if 0 <= rr < h and 0 <= cc < w and I[rr, cc] != bgc:
                        ok = False
                        break
                if not ok:
                    continue
                filled = frozenset((r0 + r, c0 + c) for (r, c) in body)
                if filled in seen_sets:
                    continue
                seen_sets.add(filled)
                sites.append({
                    'r0': r0, 'c0': c0, 'h': UH, 'w': UW,
                    'body': sorted(filled),
                    'marks': sorted((r0 + r, c0 + c) for (r, c) in mset),
                })

    if not sites:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # ---- 4) pick the route, from the input only.
    #      sweep: where the eye starts and which way it travels over the copies.
    #      stroke: fill a copy whole, or arm by arm working out from its markers.
    sig = (h * 31 + w * 17 + int(bgc) * 7 + int(fgc) * 13
           + tr0 * 5 + tc0 * 3 + BH * 11 + len(sites) * 101
           + sum(s['r0'] + 2 * s['c0'] + len(s['body']) for s in sites))
    sweep = sig % 6
    arm_by_arm = (sig // 6) % 2 == 1

    kr, kc = (tr0 + tr1) / 2.0, (tc0 + tc1) / 2.0

    def from_key(s):
        return max(abs(s['r0'] + (s['h'] - 1) / 2.0 - kr),
                   abs(s['c0'] + (s['w'] - 1) / 2.0 - kc))

    if sweep == 0:        # reading order: row by row, left to right
        sites.sort(key=lambda s: (s['r0'], s['c0']))
    elif sweep == 1:      # column by column, left to right
        sites.sort(key=lambda s: (s['c0'], s['r0']))
    elif sweep == 2:      # bottom-up, right to left
        sites.sort(key=lambda s: (-s['r0'], -s['c0']))
    elif sweep == 3:      # outward from the key, nearest copy first
        sites.sort(key=lambda s: (from_key(s), s['r0'], s['c0']))
    elif sweep == 4:      # inward, the farthest copy first
        sites.sort(key=lambda s: (-from_key(s), s['r0'], s['c0']))
    else:                 # the big copies first, then the small ones
        sites.sort(key=lambda s: (-len(s['body']), s['r0'], s['c0']))

    def arms(cellset):
        """the copy's missing body split into its 4-connected arms, each arm
        taken in turn starting with the one growing off a marker."""
        left = set(cellset)
        out = []
        while left:
            start = min(left)
            stack, blob = [start], []
            left.discard(start)
            while stack:
                r, c = stack.pop()
                blob.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (r + dr, c + dc) in left:
                        left.discard((r + dr, c + dc))
                        stack.append((r + dr, c + dc))
            out.append(sorted(blob))
        return out

    # ---- 5) paint the body back into each copy
    for s in sites:
        if arm_by_arm:
            pieces = arms(s['body'])
            pieces.sort(key=lambda blob: (
                min(max(abs(r - mr), abs(c - mc))
                    for (r, c) in blob for (mr, mc) in s['marks']),
                blob[0]))
        else:
            pieces = [s['body']]
        for piece in pieces:
            ops.append(int(fgc))
            sels.append(sel_of(piece))

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
