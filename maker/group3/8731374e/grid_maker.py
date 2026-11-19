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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    cols = list(range(10))
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    inh = random.randint(5, h - 2)
    inw = random.randint(5, w - 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw)))
    num = min(num, inh - 2, inw - 2)
    mat = [[bgc] * (inw - 2) for _ in range(inh - 2)]
    for k in range(num):
        mat[k][k] = fgc
    random.shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    random.shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    sgi = [[bgc] * inw for _ in range(inh)]
    for i in range(inh - 2):
        for j in range(inw - 2):
            sgi[i + 1][j + 1] = mat[i][j]
    go = [row[:] for row in sgi]
    dots = [(i, j) for i in range(inh) for j in range(inw) if sgi[i][j] == fgc]
    for (i, j) in dots:
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
    a, b = loci, locj
    c, d = loci + inh - 1, locj + inw - 1
    remcols = [x for x in ccols if x != bgc]
    if not remcols:
        remcols = [x for x in cols if x != bgc]
    p1 = (a - 1, random.randint(b, d))
    p2 = (random.randint(a, c), b - 1)
    p3 = (c + 1, random.randint(b, d))
    p4 = (random.randint(a, c), d + 1)
    for (i, j) in (p1, p2, p3, p4):
        gi[i][j] = random.choice(remcols)
    return {"input": gi, "output": go}


def _largest_component(I):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    best = None
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = I[r, c]
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if best is None or len(comp) > len(best[1]):
                best = (int(col), comp)
    return best


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)

    # 1. The framed rectangle: largest single-colour blob, its colour is the canvas colour.
    bgc, cells = _largest_component(I)
    top = min(p[0] for p in cells); bot = max(p[0] for p in cells)
    left = min(p[1] for p in cells); right = max(p[1] for p in cells)

    # 2. Peel off edge lines that are mostly noise (a genuine border line is canvas-coloured).
    changed = True
    while changed:
        changed = False
        for side in range(4):
            if bot - top < 2 or right - left < 2:
                break
            if side == 0:
                line = I[top, left:right + 1]
            elif side == 1:
                line = I[top:bot + 1, right]
            elif side == 2:
                line = I[bot, left:right + 1]
            else:
                line = I[top:bot + 1, left]
            if len(line) - 2 > int(np.count_nonzero(line == bgc)):
                if side == 0:
                    top += 1
                elif side == 1:
                    right -= 1
                elif side == 2:
                    bot -= 1
                else:
                    left += 1
                changed = True

    rect = I[top:bot + 1, left:right + 1]
    ho, wo = rect.shape
    vals, counts = np.unique(rect, return_counts=True)
    fgc = int(vals[np.argmin(counts)])

    dots = [(r, c) for r in range(ho) for c in range(wo) if rect[r, c] == fgc]
    dot_rows = sorted({r for r, _ in dots})
    dot_cols = sorted({c for _, c in dots})

    ops, sels = [], []
    # Crop the canvas down to the framed rectangle (full rectangle).
    ops.append(33); sels.append([top, left, ho - 1, wo - 1])

    # Every row holding a dot becomes a full line of the dot colour (all at once).
    row_cells = [(r, c) for r in dot_rows for c in range(wo)]
    if row_cells:
        ops.append(fgc); sels.append(sel_of(row_cells))
    # Every column holding a dot becomes a full line of the dot colour (all at once).
    col_cells = [(r, c) for c in dot_cols for r in range(ho)]
    if col_cells:
        ops.append(fgc); sels.append(sel_of(col_cells))

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
