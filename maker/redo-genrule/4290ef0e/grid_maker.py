"""
ARC Task: 4290ef0e (RE-ARC) — LLM-generated grid_maker
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


# --------------------------------------------------------------------------
# The task rule (re-implementation of verify_4290ef0e), shared by generate()
# and derive_operations().  Each colour in the input is one "ring" object of a
# concentric square; rings are ordered by (bbox extent + widest component),
# oriented so their corner opens down-right, nested at offsets (i, i), and the
# whole stack is stamped in all four rotations onto a (2k-1)x(2k-1) canvas.
# --------------------------------------------------------------------------

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _normalize(cs):
    r0 = min(r for r, _ in cs)
    c0 = min(c for _, c in cs)
    return frozenset((r - r0, c - c0) for r, c in cs)


def _components(cs):
    cs = set(cs)
    seen, out = set(), []
    for cell in sorted(cs):
        if cell in seen:
            continue
        st, comp = [cell], [cell]
        seen.add(cell)
        while st:
            r0, c0 = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r0 + dr, c0 + dc)
                if nb in cs and nb not in seen:
                    seen.add(nb)
                    st.append(nb)
                    comp.append(nb)
        out.append(comp)
    return out


def _rule(I):
    """Returns (ring colours outermost->innermost, predicted output, ambiguous?)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt = Counter(I.reshape(-1).tolist())
    ranked = cnt.most_common()
    bgc = ranked[0][0]
    ambiguous = len(ranked) > 1 and ranked[1][1] == ranked[0][1]
    colors = [c for c in sorted(cnt) if c != bgc]
    if not colors:
        return [], I.copy(), True

    cells = {c: [(r, cc) for r in range(h) for cc in range(w) if I[r, cc] == c]
             for c in colors}

    keys = {}
    for c in colors:
        cs = cells[c]
        rs = [r for r, _ in cs]
        cl = [x for _, x in cs]
        bbm = max(max(rs) - min(rs) + 1, max(cl) - min(cl) + 1)
        mw = max(max(x[1] for x in comp) - min(x[1] for x in comp) + 1
                 for comp in _components(cs))
        keys[c] = bbm + mw
    if len(set(keys.values())) != len(colors):
        ambiguous = True                       # tie -> DSL order() is arbitrary
    ordered = sorted(colors, key=lambda c: (-keys[c], c))

    ncol = len(colors)
    has_unit = any(len(cells[c]) == 1 for c in colors)
    k = ncol if has_unit else ncol + 1
    n = 2 * k - 1

    placed = []
    for i, c in enumerate(ordered):
        if i >= k:
            continue
        s = _normalize(cells[c])
        H = max(r for r, _ in s) + 1
        W = max(x for _, x in s) + 1
        variants = [
            s,                                                # identity
            frozenset((r, W - 1 - x) for r, x in s),          # vmirror
            frozenset((W - 1 - x, H - 1 - r) for r, x in s),  # cmirror
            frozenset((H - 1 - r, x) for r, x in s),          # hmirror
        ]
        best, bs, tied = None, -1, []
        for cd in variants:
            cdn = _normalize(cd)
            sc = int((1, 0) in cdn) + int((0, 1) in cdn)
            if sc > bs:
                bs, best, tied = sc, cdn, [cdn]
            elif sc == bs and cdn not in tied:
                tied.append(cdn)
        if len(tied) > 1:
            ambiguous = True                   # tie -> DSL argmax is arbitrary
        placed.append((c, frozenset((r + i, x + i) for r, x in best)))

    occupied = set()
    for _, st in placed:
        if occupied & st:
            ambiguous = True                   # overlapping paint order arbitrary
        occupied |= st

    g = np.full((n, n), bgc, dtype=int)

    def _paint(grid):
        for col, st in placed:
            for (r, x) in st:
                if 0 <= r < n and 0 <= x < n:
                    grid[r, x] = col
        return grid

    g = _paint(g)
    for _ in range(3):
        g = np.ascontiguousarray(np.rot90(g, k=-1))   # DSL rot90 == clockwise
        g = _paint(g)
    return ordered, g, ambiguous


# ------------------------------- 1. sample_colors ---------------------------

_VARIANTS = [{"has_center": True}, {"has_center": False}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        ex = [dict(v) for v in _VARIANTS]
        ex += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(_VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


# --------------------------------- 2. generate ------------------------------

def generate(diff_lb, diff_ub, max_h, max_w, bgc, has_center=None) -> dict:
    cols = list(range(10))
    d_ub = max(2, min(7, min(max_h, max_w) // 4))
    while True:
        hc = random.choice((True, False)) if has_center is None else bool(has_center)

        d = _unifint(diff_lb, diff_ub, (2, d_ub))
        n = 2 * d + 1
        fullh = _unifint(diff_lb, diff_ub, (min(4 * d, max_h), max_h))
        fullw = _unifint(diff_lb, diff_ub, (min(4 * d, max_w), max_w))
        if fullh < n or fullw < n:
            continue

        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, d)

        # nested corner quadrant: colour idx draws an L of length linlen at (idx,idx)
        quad = [[bgc] * (d + 1) for _ in range(d + 1)]
        for idx, c in enumerate(ccols):
            linlen = random.randint(2, d - idx + 1)
            for kk in range(linlen):
                quad[idx + kk][idx] = c
                quad[idx][idx + kk] = c

        # mirror the quadrant out into the full concentric square
        half = [[bgc] * (2 * d + 1) for _ in range(d + 1)]
        for r in range(d + 1):
            for c in range(d + 1):
                half[r][c] = quad[r][c]
        vq = [row[::-1] for row in quad]
        for r in range(d + 1):
            for c in range(d + 1):
                half[r][d + c] = vq[r][c]
        go = [row[:] for row in half] + [row[:] for row in half[::-1][1:]]

        if hc:
            others = [c for c in remcols if c not in ccols]
            go[d][d] = random.choice(others)

        # scatter every colour-object (whole ring, possibly clipped) on a canvas
        objs = {}
        for r in range(n):
            for c in range(n):
                v = go[r][c]
                if v != bgc:
                    objs.setdefault(v, []).append((r, c))
        order_objs = sorted(
            objs.items(),
            key=lambda kv: max(x[1] for x in kv[1]) - min(x[1] for x in kv[1]) + 1)

        gi = [[bgc] * fullw for _ in range(fullh)]
        fullinds = {(r, c) for r in range(fullh) for c in range(fullw)}
        inds = set(fullinds)
        ok = True
        for col, ocells in order_objs:
            objn = sorted(_normalize(ocells))
            dd = max(x[1] for x in ocells) - min(x[1] for x in ocells) + 1
            dh = max(0, dd // 2 - 1)
            base = [(r, c) for r in range(max(0, fullh - dd + 1))
                    for c in range(max(0, fullw - dd + 1))]
            cands = set(base)
            for sr, sc in ((-dh, 0), (0, -dh), (dh, 0), (0, dh)):
                cands |= {(r + sr, c + sc) for r, c in base}
            cands = sorted(cands)
            if not cands:
                ok = False
                break
            succ = False
            for _ in range(10):
                loc = random.choice(cands)
                sh = {(r + loc[0], c + loc[1]) for r, c in objn}
                vis = sh & fullinds
                if vis and vis <= inds:
                    succ = True
                    break
            if not succ:
                ok = False
                break
            for (r, c) in sh:
                if 0 <= r < fullh and 0 <= c < fullw:
                    gi[r][c] = col
            inds -= sh
        if not ok:
            continue

        gia = np.array(gi, dtype=int)
        goa = np.array(go, dtype=int)
        ordered, pred, amb = _rule(gia)
        # accept only instances the task's own rule reproduces unambiguously
        if amb or pred.shape != goa.shape or not np.array_equal(pred, goa):
            continue
        if len(set(int(v) for v in gia.reshape(-1))) - 1 != len(objs):
            continue
        return {"input": gia.tolist(), "output": goa.tolist()}


# ----------------------------- 3. derive_operations -------------------------

def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.reshape(-1).tolist()).most_common(1)[0][0])

    ordered, P, _amb = _rule(I)
    if P.shape != O.shape or not np.array_equal(P, O):   # safety net only
        P = O
        present = [c for c in sorted(set(O.reshape(-1).tolist())) if c != bgc]

        def _ext(c):
            cs = [(r, cc) for r in range(O.shape[0]) for cc in range(O.shape[1])
                  if O[r, cc] == c]
            rs = [r for r, _ in cs]
            cl = [x for _, x in cs]
            return max(max(rs) - min(rs), max(cl) - min(cl))
        ordered = sorted(present, key=lambda c: -_ext(c))

    n = P.shape[0]

    # 1. shrink the canvas to an n x n working window (the busiest one, so the
    #    clear below always has something to clear).  Full-rectangle selection:
    #    a Resize acts on the whole rectangle by design.
    best_rc, best_cnt = (0, 0), -1
    for r in range(h - n + 1):
        for c in range(w - n + 1):
            cnt = int(np.count_nonzero(I[r:r + n, c:c + n] != bgc))
            if cnt > best_cnt:
                best_cnt, best_rc = cnt, (r, c)
    R, C = best_rc

    ops, sels = [], []
    ops.append(33); sels.append([R, C, n - 1, n - 1])

    # 2. lay the background base over the whole new canvas.
    #    Full-rectangle selection: the entire canvas is the base layer.
    ops.append(bgc); sels.append([0, 0, n - 1, n - 1])

    # 3. draw the concentric rings on top, outermost ring first.
    drawn = set()
    for col in ordered:
        cells = [(r, c) for r in range(n) for c in range(n) if P[r, c] == col]
        if not cells:
            continue
        ops.append(int(col)); sels.append(sel_of(cells))
        drawn.add(int(col))
    for col in sorted(set(int(v) for v in P.reshape(-1))):
        if col == bgc or col in drawn:
            continue
        cells = [(r, c) for r in range(n) for c in range(n) if P[r, c] == col]
        ops.append(int(col)); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
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
                                f"for task 4290ef0e"
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
                    f"Failed to build a complete episode for task 4290ef0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4290ef0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
