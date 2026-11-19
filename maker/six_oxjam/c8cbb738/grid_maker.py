"""
ARC Task: c8cbb738 (RE-ARC) — LLM-generated grid_maker
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

# Reading check against the code-golf entry: the golf entry treats every colour as one rigid
# unit and pins it so its centre lands on one common centre. That works in the original ARC
# set, where every piece is symmetric. In RE-ARC the pieces are random subsets of the border,
# so the general form of that idea is "fit the piece to the border ring, and break ties by
# distance to the frame centre". The verifier and the maker already do exactly that. Both
# readings agree on the unit (one colour = one rigid piece), on what stays fixed (the frame
# corners), and on the output size (the tallest and widest piece). Nothing changes here: each
# piece slides as a whole onto the frame, and then the canvas is cropped.


def _unif(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, a + round(d * (b - a))))


def _fit(norm, oh, ow, gh, gw):
    ring = {(r, q) for r in range(gh) for q in range(gw)
            if r in (0, gh - 1) or q in (0, gw - 1)}
    K = 2 * max(gh, gw)
    cr, cc = gh // 2, gw // 2
    best, bshift = None, None
    for i in range(gh):
        for j in range(gw):
            sh = {(a + i, b + j) for a, b in norm}
            sc = K * len(sh & ring) - (abs(i + oh // 2 - cr) + abs(j + ow // 2 - cc))
            if best is None or sc > best:
                best, bshift = sc, (i, j)
    return bshift


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    gh = _unif(diff_lb, diff_ub, (3, max(3, min(10, max_h // 2))))
    gw = _unif(diff_lb, diff_ub, (3, max(3, min(10, max_w // 2))))
    h = _unif(diff_lb, diff_ub, (gh * 2, max_h))
    w = _unif(diff_lb, diff_ub, (gw * 2, max_w))
    remcols = [c for c in range(10) if c != bgc]
    ncols = _unif(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, ncols)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * gw for _ in range(gh)]
    ring = {(r, q) for r in range(gh) for q in range(gw)
            if r in (0, gh - 1) or q in (0, gw - 1)}
    crns = {(0, 0), (0, gw - 1), (gh - 1, 0), (gh - 1, gw - 1)}
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    for r, q in crns:
        go[r][q] = sc
    loci = random.randint(0, h - gh)
    locj = random.randint(0, w - gw)
    for r, q in crns:
        gi[r + loci][q + locj] = sc
    ccols = ccols[1:]
    sides = [
        ({(r, 0) for r in range(gh)}, {(0, q) for q in range(gw)}),
        ({(r, 0) for r in range(gh)}, {(gh - 1, q) for q in range(gw)}),
        ({(r, gw - 1) for r in range(gh)}, {(0, q) for q in range(gw)}),
        ({(r, gw - 1) for r in range(gh)}, {(gh - 1, q) for q in range(gw)}),
    ]
    for c in ccols:
        if len(remring) < 3:
            break
        n = _unif(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))
        obj = set(random.sample(sorted(remring), n))
        if not any(obj & b1 and obj & b2 for b1, b2 in sides):
            continue
        r0 = min(r for r, _ in obj); c0 = min(q for _, q in obj)
        oh = max(r for r, _ in obj) - r0 + 1; ow = max(q for _, q in obj) - c0 + 1
        norm = [(r - r0, q - c0) for r, q in obj]
        # keep only pieces whose rule-placement (best border fit, nearest the centre)
        # is where the generator put them -- otherwise the output is not learnable
        if _fit(norm, oh, ow, gh, gw) != (r0, c0):
            continue
        cands = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
                 if all(gi[i + a][j + b] == bgc for a, b in norm)]
        if not cands:
            continue
        i, j = random.choice(cands)
        for a, b in norm:
            gi[i + a][j + b] = c
        for r, q in obj:
            go[r][q] = c
        remring -= obj
    flat = [v for row in gi for v in row]
    if Counter(flat).most_common(1)[0][0] != bgc:
        raise ValueError("background not majority")
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """The four lone cells of one colour mark the corners of a gh x gw frame, where gh and gw
    are the height of the tallest piece and the width of the widest. Every other colour is a
    rigid piece of that frame's border. Each piece slides, unchanged, to the place on the
    border it covers best; ties go to the place nearest the frame centre, which is the golf
    entry's 'common centre' idea. Then the canvas is cropped to the frame. Only I is read."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    icols = sorted(int(v) for v in np.unique(I) if int(v) != bgc)
    cells = {c: sorted((int(r), int(q)) for r, q in zip(*np.where(I == c))) for c in icols}

    def bbox(cs):
        rs = [r for r, _ in cs]
        qs = [q for _, q in cs]
        return min(rs), min(qs), max(rs) - min(rs) + 1, max(qs) - min(qs) + 1

    gh = max(bbox(cells[c])[2] for c in icols)
    gw = max(bbox(cells[c])[3] for c in icols)

    fcol, R0, C0 = None, 0, 0
    for c in icols:
        r0, c0, hh, ww = bbox(cells[c])
        if len(cells[c]) == 4 and (hh, ww) == (gh, gw) and set(cells[c]) == {
                (r0, c0), (r0, c0 + ww - 1), (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}:
            fcol, R0, C0 = c, r0, c0
            break

    dest = {}
    for c in icols:
        if c == fcol:
            continue
        r0, c0, oh, ow = bbox(cells[c])
        norm = [(r - r0, q - c0) for r, q in cells[c]]
        bi, bj = _fit(norm, oh, ow, gh, gw)
        dest[c] = [(R0 + bi + a, C0 + bj + b) for a, b in norm]

    def in_win(p):
        return R0 <= p[0] < R0 + gh and C0 <= p[1] < C0 + gw

    grid = I.copy()
    cur = {c: list(v) for c, v in cells.items()}
    dest_cells = set(p for v in dest.values() for p in v)

    def translate(col, tgt):
        src = list(cur[col])
        dr = tgt[0][0] - src[0][0]
        dc = tgt[0][1] - src[0][1]
        if dr == 0 and dc == 0:
            return
        if col == 0:
            # ARCLE object ops only grab nonzero cells, so a 0-coloured piece cannot be
            # carried. Lift it (paint its old cells with the background) and draw it again.
            old = sorted(p for p in src if in_win(p) and p not in set(tgt))
            if old:
                ops.append(int(bgc)); sels.append(sel_of(old))
                for p in old:
                    grid[p] = bgc
            ops.append(0); sels.append(sel_of(tgt))
            for p in tgt:
                grid[p] = 0
        else:
            seq = [21 if dr > 0 else 20] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
            for k, op in enumerate(seq):
                ops.append(op)
                # first step grabs the piece; later steps keep the same grab
                sels.append(sel_of(src) if k == 0 else sel_of([]))
            for p in src:
                grid[p] = 0
            for p in tgt:
                grid[p] = col
            # only the vacated footprint reads 0; repair the part the crop keeps
            hole = sorted(p for p in set(src) - set(tgt) if in_win(p))
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
                for p in hole:
                    grid[p] = bgc
        cur[col] = sorted(tgt)

    def blockers(col):
        mine = set(cur[col])
        return {int(grid[p]) for p in dest[col] if p not in mine and grid[p] != bgc}

    pending = [c for c in icols if c in dest and sorted(dest[c]) != cur[c]]
    guard = 0
    while pending and guard < 500:
        guard += 1
        moved = False
        for col in list(pending):
            if not (blockers(col) & set(pending)):
                translate(col, sorted(dest[col]))
                pending.remove(col)
                moved = True
        if moved:
            continue
        # cyclic blocking: park one blocker on free background outside the frame
        col = pending[0]
        b = sorted(blockers(col) & set(pending))[0]
        br, bc_, bh, bw = bbox(cur[b])
        norm = [(r - br, q - bc_) for r, q in cur[b]]
        cand = None
        for rr in range(hi - bh + 1):
            for qq in range(wi - bw + 1):
                pts = [(rr + a, qq + e) for a, e in norm]
                if all(grid[p] == bgc and p not in dest_cells and not in_win(p) for p in pts):
                    d = abs(rr - br) + abs(qq - bc_)
                    if cand is None or d < cand[0]:
                        cand = (d, pts)
        if cand is None:
            break
        translate(b, sorted(cand[1]))

    # crop to the frame: the whole gh x gw window, background included
    ops.append(33); sels.append([R0, C0, gh - 1, gw - 1])
    ops.append(34); sels.append([0, 0, gh - 1, gw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
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
                                f"for task c8cbb738"
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
                    f"Failed to build a complete episode for task c8cbb738 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c8cbb738-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
