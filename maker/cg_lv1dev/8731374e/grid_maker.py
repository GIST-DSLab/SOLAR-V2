"""
ARC Task: 8731374e (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    from random import randint, choice, shuffle, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    inh = randint(5, h - 2)
    inw = randint(5, w - 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh - 2, inw - 2)))
    num = max(1, min(num, inh - 2, inw - 2))
    mat = [[bgc] * (inw - 2) for _ in range(inh - 2)]
    for k in range(num):
        mat[k][k] = fgc
    shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    sgi = [[bgc] * inw for _ in range(inh)]
    for i in range(inh - 2):
        for j in range(inw - 2):
            sgi[i + 1][j + 1] = mat[i][j]
    go = [row[:] for row in sgi]
    rows = {i for i in range(inh) for j in range(inw) if sgi[i][j] == fgc}
    cs = {j for i in range(inh) for j in range(inw) if sgi[i][j] == fgc}
    for i in range(inh):
        for j in range(inw):
            if i in rows or j in cs:
                go[i][j] = fgc
    numci = unifint(diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = sample(cols, numc)
    gi = [[choice(ccols) for _ in range(w)] for _ in range(h)]
    loci = randint(1, h - inh - 1)
    locj = randint(1, w - inw - 1)
    for i in range(inh):
        for j in range(inw):
            gi[loci + i][locj + j] = sgi[i][j]
    a, b, c, d = loci, locj, loci + inh - 1, locj + inw - 1
    remcols = [x for x in ccols if x != bgc] or [x for x in cols if x != bgc]
    gi[a - 1][randint(b, d)] = choice(remcols)
    gi[randint(a, c)][b - 1] = choice(remcols)
    gi[c + 1][randint(b, d)] = choice(remcols)
    gi[randint(a, c)][d + 1] = choice(remcols)
    return {"input": gi, "output": go}


def _largest_component_color(I):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    best, bc = 0, int(I[0, 0])
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = I[r, c]
            st = [(r, c)]
            seen[r, c] = True
            n = 0
            while st:
                y, x = st.pop()
                n += 1
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and I[yy, xx] == col:
                        seen[yy, xx] = True
                        st.append((yy, xx))
            if n > best:
                best, bc = n, int(col)
    return bc


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Canvas colour of the panel: colour of the big uniform field (largest blob).
    bgc = _largest_component_color(I)

    # The panel = the largest rectangle whose whole border is canvas colour and
    # whose inside is canvas colour plus dots of exactly ONE other colour
    # (the "mostly-uniform window" reading).  Found from I alone.
    isb = (I == bgc).astype(int)
    P = np.zeros((hi + 1, wi + 1), int)
    P[1:, 1:] = isb.cumsum(0).cumsum(1)

    def cnt(Q, r0, c0, r1, c1):
        return Q[r1 + 1, c1 + 1] - Q[r0, c1 + 1] - Q[r1 + 1, c0] + Q[r0, c0]

    colP = {}
    for k in range(10):
        m = (I == k).astype(int)
        Q = np.zeros((hi + 1, wi + 1), int)
        Q[1:, 1:] = m.cumsum(0).cumsum(1)
        colP[k] = Q

    best = None
    for r0 in range(hi):
        for r1 in range(r0 + 4, hi):
            for c0 in range(wi):
                for c1 in range(c0 + 4, wi):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best is not None and area <= best[0]:
                        continue
                    if cnt(P, r0, c0, r0, c1) != c1 - c0 + 1:
                        break
                    if cnt(P, r1, c0, r1, c1) != c1 - c0 + 1:
                        break
                    if cnt(P, r0, c0, r1, c0) != r1 - r0 + 1:
                        break
                    if cnt(P, r0, c1, r1, c1) != r1 - r0 + 1:
                        continue
                    nb = area - cnt(P, r0, c0, r1, c1)
                    if nb == 0:
                        continue
                    others = [k for k in range(10)
                              if k != bgc and cnt(colP[k], r0, c0, r1, c1) > 0]
                    if len(others) != 1:
                        continue
                    best = (area, r0, c0, r1, c1, others[0])

    _, top, left, bot, right, fgc = best
    ho, wo = bot - top + 1, right - left + 1
    rect = I[top:bot + 1, left:right + 1].copy()

    ops, sels = [], []
    # 1. Keep only the panel: crop to the whole rectangle (full-rect bbox intended).
    ops.append(33)
    sels.append([top, left, ho - 1, wo - 1])
    cur = rect.copy()
    dots = [(r, c) for r in range(ho) for c in range(wo) if rect[r, c] == fgc]

    # 2. Every row holding a dot becomes a full line of the dot colour ...
    for r in sorted({r for r, _ in dots}):
        cells = [(r, c) for c in range(wo)]
        if all(cur[y, x] == fgc for y, x in cells):
            continue
        ops.append(fgc)
        sels.append(sel_of(cells))
        cur[r, :] = fgc

    # 3. ... then every column holding a dot.
    for c in sorted({c for _, c in dots}):
        cells = [(r, c) for r in range(ho)]
        if all(cur[y, x] == fgc for y, x in cells):
            continue
        ops.append(fgc)
        sels.append(sel_of(cells))
        cur[:, c] = fgc

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8731374e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8731374e"
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
                                f"for task 8731374e"
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
                    f"Failed to build a complete episode for task 8731374e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8731374e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
