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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or fgc is None:
        bgc, fgc = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
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
    if len(remcols) == 0:
        remcols = remove(bgc, cols)
    fixobj = {
        (choice(remcols), p1), (choice(remcols), p2),
        (choice(remcols), p3), (choice(remcols), p4)
    }
    gi = paint(gi, fixobj)
    return {'input': gi, 'output': go}


def _largest_component_color(I):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    best_n, best_col = -1, None
    for i in range(h):
        for j in range(w):
            if seen[i, j]:
                continue
            col = I[i, j]
            seen[i, j] = True
            q = deque([(i, j)])
            n = 0
            while q:
                r, c = q.popleft()
                n += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if n > best_n:
                best_n, best_col = n, int(col)
    return best_col


def _find_card(I, bgc):
    """Largest rectangle (>=5x5) whose 1-cell frame is entirely the canvas colour
    and whose interior holds the canvas colour plus exactly one dot colour."""
    h, w = I.shape
    B = (I == bgc).astype(int)
    P = np.zeros((h + 1, w + 1), int)
    P[1:, 1:] = B.cumsum(0).cumsum(1)

    def cnt(r0, c0, r1, c1):  # inclusive
        if r1 < r0 or c1 < c0:
            return 0
        return P[r1 + 1, c1 + 1] - P[r0, c1 + 1] - P[r1 + 1, c0] + P[r0, c0]

    cands = []
    for r0 in range(h):
        for r1 in range(r0 + 4, h):
            for c0 in range(w):
                if not B[r0, c0] or not B[r1, c0]:
                    continue
                for c1 in range(c0 + 4, w):
                    hh, ww = r1 - r0 + 1, c1 - c0 + 1
                    if cnt(r0, c0, r0, c1) != ww:
                        break  # top edge broken: any wider rect also broken
                    if cnt(r1, c0, r1, c1) != ww:
                        break
                    if cnt(r0, c0, r1, c0) != hh or cnt(r0, c1, r1, c1) != hh:
                        continue
                    cands.append((hh * ww, r0, c0, r1, c1))
    cands.sort(reverse=True)
    for _, r0, c0, r1, c1 in cands:
        inner = I[r0 + 1:r1, c0 + 1:c1]
        others = set(inner[inner != bgc].tolist())
        if len(others) == 1:
            return r0, c0, r1, c1, others.pop()
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)

    # 1. The card's canvas colour is the colour of the largest blob in the noise.
    bgc = _largest_component_color(I)
    # 2. The card: biggest rectangle framed by canvas colour, dotted with one colour.
    #    Found from I alone (never by matching against O).
    top, left, bot, right, fgc = _find_card(I, bgc)
    ho, wo = bot - top + 1, right - left + 1
    card = I[top:bot + 1, left:right + 1]

    ops, sels = [], []
    # 3. Cut the card out of the noise (the whole card rectangle is intended).
    ops.append(33)
    sels.append([top, left, ho - 1, wo - 1])

    # 4. Lines grouped by kind: every row holding a dot becomes a full
    #    horizontal line, then every column holding a dot a full vertical line.
    #    The card frame never holds a dot, so each line visibly changes the grid.
    dot_rows = sorted({r for r in range(ho) if np.any(card[r] == fgc)})
    dot_cols = sorted({c for c in range(wo) if np.any(card[:, c] == fgc)})
    for r in dot_rows:
        ops.append(fgc)
        sels.append(sel_of([(r, c) for c in range(wo)]))
    for c in dot_cols:
        ops.append(fgc)
        sels.append(sel_of([(r, c) for r in range(ho)]))

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
