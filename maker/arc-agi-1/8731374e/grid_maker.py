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
import numpy as np
from collections import deque


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    inh = randint(5, h - 2)
    inw = randint(5, w - 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw)))
    mat = canvas(bgc, (inh - 2, inw - 2))
    tol = lambda g: list(list(e) for e in g)
    tot = lambda g: tuple(tuple(e) for e in g)
    mat = fill(mat, fgc, connect((0, 0), (num - 1, num - 1)))
    mat = tol(mat)
    shuffle(mat)
    mat = tol(dmirror(tot(mat)))
    shuffle(mat)
    mat = dmirror(tot(mat))
    sgi = paint(canvas(bgc, (inh, inw)), shift(asobject(mat), (1, 1)))
    inds = ofcolor(sgi, fgc)
    lins = mapply(fork(combine, vfrontier, hfrontier), inds)
    go = fill(sgi, fgc, lins)
    numci = unifint(diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = asindices(c)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(c, obj)
    loci = randint(1, h - inh - 1)
    locj = randint(1, w - inw - 1)
    loc = (loci, locj)
    plcd = shift(asobject(sgi), loc)
    gi = paint(gi, plcd)
    a, b = ulcorner(plcd)
    c, d = lrcorner(plcd)
    p1 = choice(totuple(connect((a - 1, b), (a - 1, d))))
    p2 = choice(totuple(connect((a, b - 1), (c, b - 1))))
    p3 = choice(totuple(connect((c + 1, b), (c + 1, d))))
    p4 = choice(totuple(connect((a, d + 1), (c, d + 1))))
    remcols = remove(bgc, ccols)
    fixobj = {
        (choice(remcols), p1), (choice(remcols), p2),
        (choice(remcols), p3), (choice(remcols), p4)
    }
    gi = paint(gi, fixobj)
    return {'input': gi, 'output': go}


def _largest_component(I):
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = I[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if best is None or len(cells) > len(best[1]):
                best = (int(col), cells)
    return best


def _dot_color(win):
    vals, cnts = np.unique(win, return_counts=True)
    return int(vals[int(np.argmin(cnts))])


def _crossfill(win):
    if win.size == 0:
        return win
    fg = _dot_color(win)
    out = win.copy()
    hh, ww = win.shape
    for r in range(hh):
        for c in range(ww):
            if win[r, c] == fg:
                out[r, :] = fg
                out[:, c] = fg
    return out


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # 1. Locate the solid rectangle: biggest same-color blob in the noise.
    bgc, cells = _largest_component(I)
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    top, bot, left, right = min(rs), max(rs), min(cs), max(cs)

    # 2. Peel border lines that random noise of the same color glued onto it:
    #    a genuine rectangle border line is (almost) entirely canvas-colored.
    for _ in range(8):
        for side in range(4):
            if bot - top < 2 or right - left < 2:
                break
            if side == 0:
                line = I[top:bot + 1, left]
            elif side == 1:
                line = I[bot, left:right + 1]
            elif side == 2:
                line = I[top:bot + 1, right]
            else:
                line = I[top, left:right + 1]
            if len(line) - 2 > int(np.count_nonzero(line == bgc)):
                if side == 0:
                    left += 1
                elif side == 1:
                    bot -= 1
                elif side == 2:
                    right -= 1
                else:
                    top += 1

    rect = I[top:bot + 1, left:right + 1]
    if rect.shape != (ho, wo) or not np.array_equal(_crossfill(rect), O):
        for r in range(hi - ho + 1):
            for c in range(wi - wo + 1):
                win = I[r:r + ho, c:c + wo]
                if np.array_equal(_crossfill(win), O):
                    top, left = r, c
                    bot, right = r + ho - 1, c + wo - 1
                    break
            else:
                continue
            break
        rect = I[top:bot + 1, left:right + 1]

    fgc = _dot_color(rect)

    ops, sels = [], []

    # 3. Keep only the rectangle: crop the canvas to it.
    ops.append(33)
    sels.append([top, left, ho - 1, wo - 1])

    # 4. Each dot shoots out its full row and full column.
    done_r, done_c = set(), set()
    for r in range(ho):
        for c in range(wo):
            if rect[r, c] != fgc:
                continue
            if r not in done_r:
                ops.append(fgc)
                sels.append([r, 0, 0, wo - 1])
                done_r.add(r)
            if c not in done_c:
                ops.append(fgc)
                sels.append([0, c, ho - 1, 0])
                done_c.add(c)

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
