"""
ARC Task: 1b2d62fb (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback only
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# 1) colours / episode plan
# ----------------------------------------------------------------------------
# The only discrete structural variant the generator has is `dmirror`:
# the separating bar is either a COLUMN (halves side by side) or a ROW
# (halves stacked).  Both must be shown to the solver.
_VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    # bgc is hardcoded to 0 in the generator -> not sampled here.
    # 8 is the answer colour, so it is excluded from the palette (as in RE-ARC);
    # 0 is the background, so no carried role ever holds colour 0.
    pool = [c for c in range(10) if c not in (0, 8)]
    barcol = random.choice(pool)
    rem = [c for c in pool if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        examples = [dict(v) for v in _VARIANTS]
        examples += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2) generator
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, barcol, cola, colb, transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice(_VARIANTS)["transposed"]

    # unmirrored layout is  h x (2w+1)
    if transposed:
        h_cap = min(30, max_w)
        w_cap = min(14, (max_h - 1) // 2)
    else:
        h_cap = min(30, max_h)
        w_cap = min(14, (max_w - 1) // 2)
    h_cap = max(2, h_cap)
    w_cap = max(2, w_cap)

    h = _unifint(diff_lb, diff_ub, (2, h_cap))
    w = _unifint(diff_lb, diff_ub, (2, w_cap))

    mp = (h * w) // 2
    deva = _unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    devb = _unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)

    inds = [(r, c) for r in range(h) for c in range(w)]
    a = random.sample(inds, numa)
    b = random.sample(inds, numb)

    gia = np.zeros((h, w), dtype=int)
    for r, c in a:
        gia[r, c] = cola
    gib = np.zeros((h, w), dtype=int)
    for r, c in b:
        gib[r, c] = colb

    gbar = np.full((h, 1), barcol, dtype=int)
    gi = np.concatenate([gia, gbar, gib], axis=1)
    go = np.where((gia == 0) & (gib == 0), 8, 0)

    if transposed:                       # dmirror == transpose
        gi = gi.T
        go = go.T

    return {"input": gi.tolist(), "output": go.tolist()}


# ----------------------------------------------------------------------------
# 3) derive_operations
# ----------------------------------------------------------------------------
_THRESH = 12          # more pieces than this -> sweep them all in one go


def _components(mask, colors=None):
    """4-connected components of `mask`, optionally split by colour."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            col = None if colors is None else colors[r, c]
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx] \
                            and (col is None or colors[ny, nx] == col):
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(comp)
    return out


def _order(regs, sweep):
    if sweep == 0:                                   # top-down, raster
        return sorted(regs, key=lambda reg: min(reg))
    if sweep == 1:                                   # bottom-up
        return sorted(regs, key=lambda reg: max(reg), reverse=True)
    if sweep == 2:                                   # biggest patch first
        return sorted(regs, key=lambda reg: (-len(reg), min(reg)))
    return sorted(regs, key=lambda reg: (min(c for _, c in reg),   # column by column
                                         min(r for r, _ in reg)))


def _sim(I, ops, cells_list):
    """Minimal ARCLE simulator for the ops used here."""
    src = np.asarray(I, dtype=int)
    G = src.copy()
    clip = None
    for op, cs in zip(ops, cells_list):
        if op == 34:
            break
        h, w = G.shape
        if 0 <= op <= 9:
            for (r, c) in cs:
                if 0 <= r < h and 0 <= c < w:
                    G[r, c] = op
        elif 10 <= op <= 19:
            if len(cs) != 1:
                continue
            r, c = cs[0]
            if not (0 <= r < h and 0 <= c < w):
                continue
            n = op - 10
            srcv = G[r, c]
            if srcv == n:
                continue
            q = deque([(r, c)])
            G[r, c] = n
            while q:
                y, x = q.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and G[ny, nx] == srcv:
                        G[ny, nx] = n
                        q.append((ny, nx))
        elif op in (28, 29):
            if not cs:
                continue
            rs = [r for r, _ in cs]
            csx = [c for _, c in cs]
            base = src if op == 28 else G
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(csx), max(csx)
            if r1 >= base.shape[0] or c1 >= base.shape[1]:
                continue
            clip = base[r0:r1 + 1, c0:c1 + 1].copy()
        elif op == 30:
            if clip is None or not cs:
                continue
            r0 = min(r for r, _ in cs)
            c0 = min(c for _, c in cs)
            for i in range(clip.shape[0]):
                for j in range(clip.shape[1]):
                    if clip[i, j] != 0 and 0 <= r0 + i < h and 0 <= c0 + j < w:
                        G[r0 + i, c0 + j] = clip[i, j]
        elif op == 33:
            if not cs:
                continue
            rs = [r for r, _ in cs]
            csx = [c for _, c in cs]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(csx), max(csx)
            G = G[r0:r1 + 1, c0:c1 + 1].copy()
    return G


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # ---- read the picture: a solid bar splits the grid into two equal halves
    vertical = (wi % 2 == 1) and len(set(I[:, wi // 2].tolist())) == 1
    if vertical:
        ho, wo = hi, (wi - 1) // 2
        pos_a, pos_b = (0, 0), (0, wo + 1)
        barcol = int(I[0, wo])
    else:
        ho, wo = (hi - 1) // 2, wi
        pos_a, pos_b = (0, 0), (ho + 1, 0)
        barcol = int(I[ho, 0])

    A = I[pos_a[0]:pos_a[0] + ho, pos_a[1]:pos_a[1] + wo]
    B = I[pos_b[0]:pos_b[0] + ho, pos_b[1]:pos_b[1] + wo]

    nA = int(np.count_nonzero(A))
    nB = int(np.count_nonzero(B))

    # ---- which half do we work in?  the one with fewer marks to clear away
    if nA != nB:
        keep_first = nA <= nB
    else:
        keep_first = (barcol % 2 == 0)
    if keep_first:
        K, kpos, M, mpos = A, pos_a, B, pos_b
    else:
        K, kpos, M, mpos = B, pos_b, A, pos_a
    kr, kc = kpos
    mr, mc = mpos

    # ---- route, chosen from the picture itself (same input -> same route)
    method = (nA + 2 * nB + barcol) % 3          # 0 stamp-first, 1 lay-8-first, 2 draw direct
    late = (ho + wo + barcol) % 2                # crop now or crop at the end
    sweep = (nA + ho) % 4                        # which piece first / which way to sweep

    ANS = (K == 0) & (M == 0)
    zero_regs = _components(K == 0)
    # stamping the other half over this one is only load-bearing when some empty
    # patch of K is *partly* (not wholly) covered by the other half's marks
    partially = any(
        any(M[r, c] != 0 for r, c in reg) and any(M[r, c] == 0 for r, c in reg)
        for reg in zero_regs
    )
    if method != 2 and not partially:
        method = 2

    merged = K.copy()
    mm = M != 0
    merged[mm] = M[mm]

    def build(method, late, sweep):
        ops, cells = [], []
        state = {"cropped": False}

        def cur(r, c):
            return (r, c) if state["cropped"] else (r + kr, c + kc)

        def emit(op, cs):
            ops.append(op)
            cells.append([(int(r), int(c)) for r, c in cs])

        def do_crop():
            # selection is exactly the full rectangle of the kept half
            emit(33, [cur(r, c) for r in range(ho) for c in range(wo)])
            state["cropped"] = True

        def do_stamp():
            # the other half is taken from the INPUT, so this works cropped or not
            emit(28, [(mr + r, mc + c) for r in range(ho) for c in range(wo)])
            emit(30, [cur(0, 0)])

        def paint8(regs, use_flood):
            regs = _order(regs, sweep)
            if not regs:
                return
            if len(regs) <= _THRESH:
                for reg in regs:
                    if use_flood:
                        r, c = reg[0]
                        emit(18, [cur(r, c)])           # FloodFill8 on one empty patch
                    else:
                        emit(8, [cur(r, c) for r, c in reg])
            else:
                emit(8, [cur(r, c) for reg in regs for (r, c) in reg])

        def erase(regs):
            regs = _order(regs, sweep)
            if not regs:
                return
            if len(regs) <= _THRESH:
                for reg in regs:
                    r, c = reg[0]
                    emit(10, [cur(r, c)])               # FloodFill0 on one mark
            else:
                emit(0, [cur(r, c) for reg in regs for (r, c) in reg])

        if method == 0:
            # stamp the other half on top, then mark what is still empty
            do_stamp()
            if not late:
                do_crop()
            paint8(_components(merged == 0), True)
            erase(_components(merged != 0, merged))
            if late:
                do_crop()
        elif method == 1:
            # lay the 8s down first, then stamp the other half over them
            base = [reg for reg in zero_regs if any(M[r, c] == 0 for r, c in reg)]
            paint8(base, True)
            do_stamp()
            if not late:
                do_crop()
            erase(_components(merged != 0, merged))
            if late:
                do_crop()
        else:
            # draw the overlap straight onto this half
            if not late:
                do_crop()
            paint8(_components(ANS), False)
            erase(_components(K != 0, K))
            if late:
                do_crop()

        emit(34, [(r, c) for r in range(ho) for c in range(wo)])
        return ops, cells

    ops, cells = build(method, late, sweep)

    if not np.array_equal(_sim(I, ops, cells), O):
        # safe fallback: crop, mark the overlap, wipe this half's marks
        ops, cells = build(2, 0, sweep)

    # ---- drop anything whose removal would not change the answer
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops) - 1:
            t_ops = ops[:i] + ops[i + 1:]
            t_cells = cells[:i] + cells[i + 1:]
            if np.array_equal(_sim(I, t_ops, t_cells), O):
                ops, cells = t_ops, t_cells
                changed = True
            else:
                i += 1

    sels = [sel_of(cs) for cs in cells]
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
                        f"num_examples+1 ({num_examples + 1}) for task 1b2d62fb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1b2d62fb"
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
                                f"for task 1b2d62fb"
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
                    f"Failed to build a complete episode for task 1b2d62fb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1b2d62fb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
