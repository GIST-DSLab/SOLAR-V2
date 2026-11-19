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
from collections import deque


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def _find_panel(I):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # 1. The framed panel's background is the largest same-colour connected blob.
    seen = np.zeros_like(I, dtype=bool)
    best = []
    for sr in range(hi):
        for sc in range(wi):
            if seen[sr, sc]:
                continue
            col = I[sr, sc]
            comp = []
            dq = deque([(sr, sc)]); seen[sr, sc] = True
            while dq:
                r, c = dq.popleft(); comp.append((r, c))
                for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True; dq.append((nr, nc))
            if len(comp) > len(best):
                best = comp
    bgc = int(I[best[0]])
    top = min(r for r, _ in best); bot = max(r for r, _ in best)
    left = min(c for _, c in best); right = max(c for _, c in best)
    # 2. Inside that blob's bounding box, the panel is the biggest rectangle
    #    whose frame lines are entirely bgc and whose inside holds bgc plus
    #    exactly one other (dot) colour.
    B = (I == bgc).astype(int)
    rowp = np.concatenate([np.zeros((hi, 1), int), np.cumsum(B, 1)], 1)
    colp = np.concatenate([np.zeros((1, wi), int), np.cumsum(B, 0)], 0)
    P = {}
    for v in range(10):
        M = (I == v).astype(int)
        P[v] = np.pad(M.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def cnt(v, r0, c0, r1, c1):
        p = P[v]
        return p[r1+1, c1+1] - p[r0, c1+1] - p[r1+1, c0] + p[r0, c0]

    cand = None
    T, Bo, L, R = top, bot, left, right
    for r0 in range(T, Bo + 1):
        for r1 in range(r0 + 4, Bo + 1):
            for c0 in range(L, R + 1):
                if B[r0, c0] == 0 or B[r1, c0] == 0:
                    continue
                for c1 in range(c0 + 4, R + 1):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if cand is not None and area <= cand[0]:
                        continue
                    if rowp[r0, c1+1] - rowp[r0, c0] != c1 - c0 + 1:
                        break
                    if rowp[r1, c1+1] - rowp[r1, c0] != c1 - c0 + 1:
                        break
                    if colp[r1+1, c0] - colp[r0, c0] != r1 - r0 + 1:
                        break
                    if colp[r1+1, c1] - colp[r0, c1] != r1 - r0 + 1:
                        continue
                    nb = cnt(bgc, r0, c0, r1, c1)
                    if nb == area:
                        continue
                    if any(cnt(v, r0, c0, r1, c1) == area - nb for v in range(10) if v != bgc):
                        cand = (area, r0, c0, r1, c1)
    if cand is None:
        return None
    _, top, left, bot, right = cand
    return bgc, top, left, bot, right


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None):
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(range(10), 2)
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    inh = random.randint(5, h - 2)
    inw = random.randint(5, w - 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw) - 2))
    mat = [[bgc] * (inw - 2) for _ in range(inh - 2)]
    for i in range(num):
        mat[i][i] = fgc
    random.shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    random.shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    sgi = [[bgc] * inw for _ in range(inh)]
    for i in range(inh - 2):
        for j in range(inw - 2):
            sgi[i + 1][j + 1] = mat[i][j]
    go = [row[:] for row in sgi]
    for i in range(inh):
        for j in range(inw):
            if sgi[i][j] == fgc:
                for jj in range(inw):
                    go[i][jj] = fgc
                for ii in range(inh):
                    go[ii][j] = fgc
    numci = unifint(diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = random.sample(cols, numc)
    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    loci = random.randint(1, h - inh - 1)
    locj = random.randint(1, w - inw - 1)
    for i in range(inh):
        for j in range(inw):
            gi[loci + i][locj + j] = sgi[i][j]
    a, b, c, d = loci, locj, loci + inh - 1, locj + inw - 1
    remcols = [x for x in ccols if x != bgc]
    gi[a - 1][random.randint(b, d)] = random.choice(remcols)
    gi[random.randint(a, c)][b - 1] = random.choice(remcols)
    gi[c + 1][random.randint(b, d)] = random.choice(remcols)
    gi[random.randint(a, c)][d + 1] = random.choice(remcols)
    found = _find_panel(gi)
    if found is None or found[1:] != (a, b, c, d) or found[0] != bgc:
        return None
    return {"input": gi, "output": go}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    for _ in range(200):
        res = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, fgc)
        if res is not None:
            return res
    raise ValueError("could not build an unambiguous panel")


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    # 1-2. Find the framed panel hidden in the noise (from I only).
    bgc, top, left, bot, right = _find_panel(I)
    rect = I[top:bot+1, left:right+1]
    ho, wo = rect.shape
    others = [int(v) for v in np.unique(rect) if v != bgc]
    fgc = others[0]
    dots = [(r, c) for r in range(ho) for c in range(wo) if rect[r, c] == fgc]

    ops, sels = [], []
    # 3. Cut the panel out of the noise (full rectangle crop).
    ops.append(33); sels.append([top, left, ho - 1, wo - 1])
    # 4. Every row holding a dot becomes a full line of the dot colour.
    rows = sorted({r for r, _ in dots})
    ops.append(fgc); sels.append(sel_of([(r, c) for r in rows for c in range(wo)]))
    # 5. Every column holding a dot becomes a full line of the dot colour.
    dcols = sorted({c for _, c in dots})
    ops.append(fgc); sels.append(sel_of([(r, c) for r in range(ho) for c in dcols]))
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
