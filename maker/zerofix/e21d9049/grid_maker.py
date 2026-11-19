"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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
import math
import random
from collections import Counter

import numpy as np


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is a real "role" in this task: the rule (repeat the
    # short bar's colour pattern along its whole row / column) is colour agnostic, so the
    # bar colours stay freshly sampled per instance -- including the colour 0.
    return {"bgc": random.choice(list(range(10)))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0) -> dict:
    def uf(lo, hi):
        if hi <= lo:
            return lo
        va = lo + (hi - lo) * diff_lb
        vb = lo + (hi - lo) * diff_ub
        a = max(lo, int(math.ceil(va)))
        b = min(hi, int(math.floor(vb)))
        if b < a:
            a, b = lo, hi
        return random.randint(a, b)

    def unique_line(a, b, x, cross_in):
        # cells of this bar are a..b ; x is the other bar's line index, and cross_in says
        # whether the crossing cell is painted by the OTHER bar.  The bar must be the only
        # readable explanation of the cells that appear on this line (otherwise the period
        # of the repetition would be ambiguous and the episode unlearnable).
        R = set(range(a, b + 1))
        if cross_in:
            R.add(x)
        n = 0
        if len(R) >= 2 and (max(R) - min(R) + 1) == len(R):
            n += 1
        if x in R:
            R2 = R - {x}
            if len(R2) >= 2 and (max(R2) - min(R2) + 1) == len(R2):
                n += 1
        return n == 1

    hlo, wlo = min(10, max_h), min(10, max_w)
    remcols = [c for c in range(10) if c != bgc]

    while True:
        h = uf(hlo, max_h)
        w = uf(wlo, max_w)
        ph = uf(2, min(9, h))
        pw = uf(2, min(9, w))
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        a, b = locih, locih + ph - 1
        c0, d0 = locjw, locjw + pw - 1
        inH = a <= lociw <= b          # crossing cell belongs to the vertical bar
        inW = c0 <= locjh <= d0        # crossing cell belongs to the horizontal bar
        if not (inH and inW):
            # either a proper crossing (shared cell) or two clearly separated bars
            drow = 0 if inH else min(abs(lociw - a), abs(lociw - b))
            dcol = 0 if inW else min(abs(locjh - c0), abs(locjh - d0))
            if max(drow, dcol) < 2:
                continue
        if not unique_line(a, b, lociw, inW):
            continue
        if not unique_line(c0, d0, locjh, inH):
            continue
        break

    vpat = [random.choice(remcols) for _ in range(ph)]
    hpat = [random.choice(remcols) for _ in range(pw)]
    col = random.choice(remcols)
    vpat[(lociw - a) % ph] = col
    hpat[(locjh - c0) % pw] = col

    gi = [[bgc] * w for _ in range(h)]
    for t in range(ph):
        gi[a + t][locjh] = vpat[t]
    for t in range(pw):
        gi[lociw][c0 + t] = hpat[t]

    go = [row[:] for row in gi]
    for r in range(h):
        go[r][locjh] = vpat[(r - a) % ph]
    for cc in range(w):
        go[lociw][cc] = hpat[(cc - c0) % pw]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O, examples=None):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # ---- background: the colour the canvas is made of (confirmed on the demonstrations) --
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ex_o in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    nz = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]

    # ---- the two lines: the column that carries the vertical bar, the row that carries
    #      the horizontal bar (each holds far more marks than any other line) -------------
    col_j = Counter(c for _r, c in nz).most_common(1)[0][0]
    row_i = Counter(r for r, _c in nz).most_common(1)[0][0]

    def pick(idxs, other):
        # the bar is the contiguous run on this line; the single cell contributed by the
        # OTHER bar (at index `other`) may sit on this line too and must be discarded.
        s = sorted(idxs)
        cands = []
        if len(s) >= 2 and s[-1] - s[0] + 1 == len(s):
            cands.append(s)
        if other in s:
            s2 = [v for v in s if v != other]
            if len(s2) >= 2 and s2[-1] - s2[0] + 1 == len(s2):
                cands.append(s2)
        return cands[0] if cands else s

    vrows = pick([r for r, c in nz if c == col_j], row_i)
    hcols = pick([c for r, c in nz if r == row_i], col_j)

    start_v, vpat = vrows[0], [int(I[r, col_j]) for r in vrows]
    start_h, hpat = hcols[0], [int(I[row_i, c]) for c in hcols]

    ops, sels = [], []
    G = I.copy()

    def emit_line(pos, n, start, pat):
        """Repeat `pat` (the bar, anchored at `start`) over the whole line.
        pos(k) -> grid coordinate of index k along that line."""
        L = len(pat)
        tgt = [pat[(k - start) % L] for k in range(n)]
        origins = [o for o in range(start % L, n - L + 1, L) if o != start]

        # --- stamp the bar itself at every full repeat that fits -----------------------
        pastes = []
        if any(v != 0 for v in pat):
            sim = G.copy()
            for o in origins:
                changed = False
                for t in range(L):
                    if pat[t] != 0:                      # 0 never travels with a Copy
                        r, c = pos(o + t)
                        if sim[r, c] != pat[t]:
                            sim[r, c] = pat[t]
                            changed = True
                if changed:
                    pastes.append(o)
        if pastes:
            ops.append(28)                               # CopyI: the bar as drawn in I
            sels.append(sel_of([pos(start + t) for t in range(L)]))
            for o in pastes:
                ops.append(30)                           # Paste at this repeat's origin
                sels.append(sel_of([pos(o)]))
                for t in range(L):
                    if pat[t] != 0:
                        r, c = pos(o + t)
                        G[r, c] = pat[t]

        # --- draw every colour stripe of the pattern that the stamps could not carry:
        #     the 0-coloured cells (Paste treats 0 as "nothing there") and the partial
        #     repeats that overrun the two ends of the line ------------------------------
        order = []
        for v in pat:
            if v not in order:
                order.append(v)
        for cval in order:
            cells = []
            for k in range(n):
                if tgt[k] == cval:
                    r, c = pos(k)
                    if G[r, c] != cval:                  # already right -> nothing to do
                        cells.append((r, c))
            if cells:
                ops.append(cval)                         # Color<cval>
                sels.append(sel_of(cells))
                for (r, c) in cells:
                    G[r, c] = cval

    emit_line(lambda k: (k, col_j), h, start_v, vpat)    # vertical bar down its column
    emit_line(lambda k: (row_i, k), w, start_h, hpat)    # horizontal bar along its row

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                    # full-grid bbox: Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
