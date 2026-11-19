"""
ARC Task: 234bbc79 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc = random.sample(cols, 2)
    ccols = [c for c in cols if c not in (bgc, dotc)]
    return {"bgc": bgc, "dotc": dotc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, ccols=None) -> dict:
    if ccols is None:
        ccols = [c for c in range(10) if c not in (bgc, dotc)]
    while True:
        h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
        w = unifint(diff_lb, diff_ub, (6, max(6, min(20, max_w - 4))))
        ncols = unifint(diff_lb, diff_ub, (1, min(8, len(ccols))))
        pcols = random.sample(list(ccols), ncols)
        go = [[bgc] * 30 for _ in range(h)]
        spi = randint(0, h - 1)
        snek = [(spi, 0)]
        go[spi][0] = dotc
        while True:
            previ, prevj = snek[-1]
            if prevj == w - 1:
                if choice((True, False, False)):
                    break
            options = []
            if previ < h - 1 and go[previ + 1][prevj] == bgc:
                options.append((previ + 1, prevj))
            if previ > 0 and go[previ - 1][prevj] == bgc:
                options.append((previ - 1, prevj))
            if prevj < w - 1:
                options.append((previ, prevj + 1))
            if len(options) == 0:
                break
            loc = choice(options)
            snek.append(loc)
            go[loc[0]][loc[1]] = dotc
        objs, cobj = [], []
        for idx, cel in enumerate(snek):
            if len(cobj) > 2 and width(frozenset(cobj)) > 1 and snek[idx - 1] == (cel[0], cel[1] - 1):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        if len(objs) == 0:
            continue
        objs[-1] += cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = unifint(diff_lb, diff_ub, (2, nobjs))
        for k in range(nobjs - ntokeep):
            idx = randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        n = len(objs)
        if w + n + 1 > max_w:
            continue
        fullw = unifint(diff_lb, diff_ub, (w + n + 1, max(w + n + 1, min(30, max_w))))
        inobjs = []
        for idx, obj in enumerate(objs):
            col = choice(pcols)
            for (i, j) in obj:
                go[i][j] = col
            centerpart = recolor(col, set(obj[1:-1]))
            leftpart = {(dotc if idx > 0 else col, obj[0])}
            rightpart = {(dotc if idx < n - 1 else col, obj[-1])}
            inobjs.append(centerpart | leftpart | rightpart)
        spacings = [1] * (n - 1)
        for k in range(fullw - w - n - 1):
            spacings[randint(0, len(spacings) - 1)] += 1
        lspacings = [0] + spacings
        gi = canvas(bgc, (h, fullw))
        ofs = 0
        for i, (lsp, obj) in enumerate(zip(lspacings, inobjs)):
            if i == 0:
                ulc = ulcorner(obj)
            else:
                ulc = (randint(0, h - height(obj)), ofs + lsp)
            ofs += width(obj) + lsp
            gi = paint(gi, shift(normalize(obj), ulc))
        break
    go = [row[:w] for row in go]
    return {"input": [list(r) for r in gi], "output": go}


def _pieces(I, bgc):
    # pieces are separated by all-background columns
    h, W = I.shape
    occ = [bool((I[:, c] != bgc).any()) for c in range(W)]
    segs, c = [], 0
    while c < W:
        if occ[c]:
            s = c
            while c < W and occ[c]:
                c += 1
            segs.append((s, c - 1))
        else:
            c += 1
    return [[(r, cc) for cc in range(a, b + 1) for r in range(h) if I[r, cc] != bgc]
            for a, b in segs]


def _dot_ok(I, pieces, d):
    n = len(pieces)
    if n < 2:
        return False
    for k, p in enumerate(pieces):
        lo = min(c for _, c in p)
        hi = max(c for _, c in p)
        m = [(r, c) for r, c in p if I[r, c] == d]
        want = {c for c, need in ((lo, k > 0), (hi, k < n - 1)) if need}
        if len(m) != len(want) or {c for _, c in m} != want:
            return False
        if len({int(I[q]) for q in p if int(I[q]) != d}) != 1:
            return False
    return True


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, wi = I.shape
    bgc = int(I[0, wi - 1])          # generator always leaves the last two columns empty
    pieces = _pieces(I, bgc)
    cands = [d for d in range(10) if d != bgc and _dot_ok(I, pieces, d)]
    if len(cands) > 1 and examples:
        for ex_i, _ in examples:
            E = np.asarray(ex_i, dtype=int)
            eb = int(E[0, -1])
            ep = _pieces(E, eb)
            cands = [d for d in cands if _dot_ok(E, ep, d)] or cands
    if not cands:
        return [34], [[0, 0, h - 1, wi - 1]]
    d = min(cands, key=lambda c: int((I == c).sum()))
    n = len(pieces)

    ops, sels = [], []
    W = sum(max(c for _, c in p) - min(c for _, c in p) + 1 for p in pieces)
    body = []            # body colour of each piece
    final = []           # final cells of each piece
    prev_right = None
    for k, p in enumerate(pieces):
        col = next(int(I[q]) for q in p if int(I[q]) != d)
        body.append(col)
        lo = min(c for _, c in p)
        hi = max(c for _, c in p)
        if k == 0:
            dr = dc = 0
        else:
            L = next(q for q in p if I[q] == d and q[1] == lo)
            dr, dc = prev_right[0] - L[0], prev_right[1] + 1 - L[1]
        if k < n - 1:
            R = next(q for q in p if I[q] == d and q[1] == hi)
            prev_right = (R[0] + dr, R[1] + dc)
        if k > 0 and col != 0:
            # slide the piece so its left marker lands just right of the previous
            # piece's right marker: one grab, then empty selections
            steps = ([(21 if dr > 0 else 20, 1 if dr > 0 else -1, 0)] * abs(dr)
                     + [(22 if dc > 0 else 23, 0, 1 if dc > 0 else -1)] * abs(dc))
            for s, (op, sr, sc) in enumerate(steps):
                ops.append(op)
                sels.append(sel_of(p) if s == 0 else sel_of([]))
            final.append([(r + dr, c + dc) for r, c in p])
        elif k > 0:
            # ARCLE never carries 0 cells, so a colour-0 piece cannot be slid: it is
            # inked at its destination by the final colour pass instead
            final.append([(r + dr, c + dc) for r, c in p])
        else:
            final.append(list(p))

    # crop the canvas to the joined chain's width (whole rectangle)
    if W < wi:
        ops.append(33)
        sels.append([0, 0, h - 1, W - 1])

    # restore background on the footprints the displaced pieces left inside the canvas
    # (slid pieces leave 0 there; a colour-0 piece's footprint is still drawn)
    occupied = {q for fc in final for q in fc}
    vacated = {q for p in pieces[1:] for q in p} - occupied
    holes = [q for q in vacated if q[1] < W]
    if bgc != 0 and holes:              # bgc 0: the vacated 0s already are background
        ops.append(bgc)
        sels.append(sel_of(holes))

    # joined markers become body cells: recolour each piece's marker cells, per colour
    groups = {}
    for k, p in enumerate(pieces):
        dr = final[k][0][0] - p[0][0]
        dc = final[k][0][1] - p[0][1]
        if body[k] == 0 and k > 0:
            cells = final[k]      # the whole colour-0 piece at its destination
        else:
            cells = [(r + dr, c + dc) for r, c in p if I[r, c] == d]
        groups.setdefault(body[k], []).extend(cells)
    for col, cells in groups.items():
        ops.append(col)
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
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
                                f"for task 234bbc79"
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
                    f"Failed to build a complete episode for task 234bbc79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"234bbc79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
