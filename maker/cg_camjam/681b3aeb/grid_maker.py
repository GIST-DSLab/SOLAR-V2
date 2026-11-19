"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ca, cb = random.sample(cols, 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi_ = a + int(round(ub * (b - a)))
        lo, hi_ = max(a, min(lo, b)), max(a, min(hi_, b))
        if hi_ < lo:
            lo, hi_ = hi_, lo
        return random.randint(lo, hi_)

    def dnb(p):
        r, c = p
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    def is_rect(S):
        rs = [r for r, _ in S]
        cs = [c for _, c in S]
        return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    def normalize(S):
        r0 = min(r for r, _ in S)
        c0 = min(c for _, c in S)
        return {(r - r0, c - c0) for r, c in S}

    if bgc is None or ca is None or cb is None:
        bgc, ca, cb = random.sample(range(10), 3)

    max_side = max(2, min(8, max_h // 3, max_w // 3))
    while True:
        hi = unifint(diff_lb, diff_ub, (2, max_side))
        h = unifint(diff_lb, diff_ub, (3 * hi, max_h))
        w = unifint(diff_lb, diff_ub, (3 * hi, max_w))
        inds = [(i, j) for i in range(hi) for j in range(hi)]
        while True:
            pa = random.choice(inds)
            rem = [x for x in inds if x != pa]
            pb = random.choice(rem)
            rem = [x for x in rem if x != pb]
            A, B = {pa}, {pb}
            remset = set(rem)
            for _ in range(len(rem)):
                acands = remset & {n for p in A for n in dnb(p)}
                bcands = remset & {n for p in B for n in dnb(p)}
                opts = []
                if acands:
                    opts.append(0)
                if bcands:
                    opts.append(1)
                idx = random.choice(opts)
                if idx == 0:
                    loc = random.choice(sorted(acands))
                    A.add(loc)
                else:
                    loc = random.choice(sorted(bcands))
                    B.add(loc)
                remset.discard(loc)
            if not (is_rect(A) or is_rect(B)):
                break
        go = [[0] * hi for _ in range(hi)]
        for r, c in A:
            go[r][c] = ca
        for r, c in B:
            go[r][c] = cb
        A = normalize(A)
        B = normalize(B)
        ha = max(r for r, _ in A) + 1
        wa = max(c for _, c in A) + 1
        hb = max(r for r, _ in B) + 1
        wb = max(c for _, c in B) + 1
        ok = False
        if ha <= h and wa <= w:
            for _ in range(10):
                li = random.randint(0, h - ha)
                lj = random.randint(0, w - wa)
                plA = {(r + li, c + lj) for r, c in A}
                rem2 = [(i, j) for i in range(h) for j in range(w)
                        if (i, j) not in plA and i <= h - hb and j <= w - wb]
                if not rem2:
                    continue
                ch = random.choice(rem2)
                plB = {(r + ch[0], c + ch[1]) for r, c in B}
                if plB.issubset(set(rem2)):
                    ok = True
                    break
        if ok:
            break
    gi = [[bgc] * w for _ in range(h)]
    for r, c in plA:
        gi[r][c] = ca
    for r, c in plB:
        gi[r][c] = cb
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading: one piece stays put and defines the rectangle; the rectangle is
    cropped around it, and every cell of that rectangle that is not the anchor
    piece is the other piece's colour (canvas of one colour, the other drawn in)."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- background colour, confirmed by the episode's example inputs
    grids = [I]
    if examples:
        for pair in examples:
            try:
                grids.append(np.asarray(pair[0], dtype=int))
            except Exception:
                pass
    votes = Counter()
    for g in grids:
        votes[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]
    if not bool((I == bgc).any()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- the two pieces
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append((col, sorted(cells)))
    if len(comps) < 2:
        return [34], [[0, 0, h - 1, w - 1]]
    comps.sort(key=lambda t: -len(t[1]))
    comps = comps[:2]          # [larger piece, smaller piece]

    pieces = []
    for col, cells in comps:
        ul = (min(r for r, _ in cells), min(c for _, c in cells))
        n = {(r - ul[0], c - ul[1]) for r, c in cells}
        pieces.append({"col": col, "cells": cells, "ul": ul, "n": n})
    P0, P1 = pieces
    nA, nB = P0["n"], P1["n"]
    span = 2 + max(max(r for r, _ in nA), max(r for r, _ in nB),
                   max(c for _, c in nA), max(c for _, c in nB))

    # ---- interlocking offset: union is a solid rectangle, no overlap
    sol = None
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            sb = {(r + dr, c + dc) for r, c in nB}
            if sb & nA:
                continue
            u = nA | sb
            r0 = min(r for r, _ in u); c0 = min(c for _, c in u)
            r1 = max(r for r, _ in u); c1 = max(c for _, c in u)
            if len(u) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            sol = (dr - r0, dc - c0, r1 - r0 + 1, c1 - c0 + 1, -r0, -c0)
            break
        if sol is not None:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]
    hr, wr = sol[2], sol[3]
    P0["p"] = (sol[4], sol[5])
    P1["p"] = (sol[0], sol[1])

    # ---- choose the anchor piece: the one whose rectangle fits where it lies
    anchor = other = None
    R0 = C0 = None
    for X, Y in ((P0, P1), (P1, P0)):
        r0 = X["ul"][0] - X["p"][0]
        c0 = X["ul"][1] - X["p"][1]
        if 0 <= r0 and r0 + hr <= h and 0 <= c0 and c0 + wr <= w:
            anchor, other, R0, C0 = X, Y, r0, c0
            break

    ops, sels = [], []
    G = I.copy()
    if anchor is None:
        # neither rectangle fits in place: carry a (non-zero) anchor inward
        cand = [(X, Y) for X, Y in ((P0, P1), (P1, P0)) if X["col"] != 0]
        anchor, other = cand[0]
        r0 = anchor["ul"][0] - anchor["p"][0]
        c0 = anchor["ul"][1] - anchor["p"][1]
        R0 = min(max(r0, 0), h - hr)
        C0 = min(max(c0, 0), w - wr)
        d = (R0 - r0, C0 - c0)
        steps = []
        steps += [(20 if d[0] < 0 else 21, (-1 if d[0] < 0 else 1, 0))] * abs(d[0])
        steps += [(23 if d[1] < 0 else 22, (0, -1 if d[1] < 0 else 1))] * abs(d[1])
        cur = list(anchor["cells"])
        first = True
        for op, (sr, sc) in steps:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + sr, c + sc) for r, c in cur]
        # simulate: footprint zeroed, path restored, piece pasted at destination
        for r, c in anchor["cells"]:
            G[r, c] = 0
        for r, c in cur:
            if 0 <= r < h and 0 <= c < w:
                G[r, c] = anchor["col"]

    # ---- crop the canvas to the rectangle around the anchor (full rect)
    ops.append(33)
    sels.append([R0, C0, hr - 1, wr - 1])
    win = G[R0:R0 + hr, C0:C0 + wr].copy()

    # ---- every rectangle cell that is not the anchor becomes the other piece's colour
    ca, co = anchor["col"], other["col"]
    fill = [(r, c) for r in range(hr) for c in range(wr)
            if win[r, c] != ca and win[r, c] != co]
    if fill:
        ops.append(co)
        sels.append(sel_of(fill))

    ops.append(34)
    sels.append([0, 0, hr - 1, wr - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
