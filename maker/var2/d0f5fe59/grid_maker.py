"""
ARC Task: d0f5fe59 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, min(30, (h * w) // 9))))
    nfound = 0
    trials = 0
    maxtrials = nobjs * 5
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while trials < maxtrials and nfound < nobjs:
        oh = unifint(diff_lb, diff_ub, (1, 5))
        ow = unifint(diff_lb, diff_ub, (1, 5))
        bx = asindices(canvas(-1, (oh, ow)))
        sp = choice(totuple(bx))
        shp = {sp}
        dev = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((dev, oh * ow - dev))
        ncells = min(max(1, ncells), oh * ow - 1)
        for k in range(ncells):
            cands = totuple((bx - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            ij = choice(cands)
            shp.add(ij)
        shp = normalize(shp)
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        plcd = shift(shp, loc)
        if plcd.issubset(inds):
            gi = fill(gi, fgc, plcd)
            inds = (inds - plcd) - mapply(neighbors, plcd)
            nfound += 1
        trials += 1
    if nfound == 0:
        loc = choice(totuple(asindices(gi)))
        gi = fill(gi, fgc, frozenset({loc}))
        nfound = 1
    go = canvas(bgc, (nfound, nfound))
    go = fill(go, fgc, connect((0, 0), (nfound - 1, nfound - 1)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read off I alone):  count the separate objects -> n.
       The answer is an n x n sheet of the picture's background colour with the
       main diagonal drawn in the picture's rare (object) colour.
       O is never consulted; the route below is decided by measurements of I."""
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- colours: background = most common colour of I, object colour = least common ---
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    rest = [c for c in cnt if c != bgc]
    fgc = min(rest, key=lambda c: (cnt[c], c)) if rest else bgc

    # --- objects: 4-connected same-colour components of the non-background cells ---
    seen = np.zeros((hi, wi), bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == I[y, x]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)

    n = max(1, min(30, len(comps)))
    nfg = int(sum(len(c) for c in comps))
    first = min(min(c) for c in comps) if comps else (0, 0)

    target = np.full((n, n), bgc, dtype=int)
    for i in range(n):
        target[i, i] = fgc

    diag = [(i, i) for i in range(n)]
    off = [(r, c) for r in range(n) for c in range(n) if r != c]

    def build(style, prime_first, downward, crop_last):
        ops, sels = [], []
        box = [I.copy()]
        states = [(box[0].shape, box[0].tobytes())]

        def snap():
            states.append((box[0].shape, box[0].tobytes()))

        def fill_rect(r0, c0, h, w, color):
            g = box[0]
            if np.any(g[r0:r0 + h, c0:c0 + w] != color):
                ops.append(int(color))
                sels.append([r0, c0, h - 1, w - 1])   # exactly this full rectangle, background included
                g[r0:r0 + h, c0:c0 + w] = color
                snap()

        def paint(cells, color):
            g = box[0]
            cells = [(int(r), int(c)) for r, c in cells]
            if not cells or all(g[rc] == color for rc in cells):
                return
            ops.append(int(color))
            sels.append(sel_of(cells))
            for rc in cells:
                g[rc] = color
            snap()

        # ---- canvas: either open the n x n sheet now, or draw in the corner and crop later
        if (not crop_last) and (hi, wi) != (n, n):
            ops.append(33)
            sels.append([0, 0, n - 1, n - 1])
            new = np.zeros((n, n), dtype=int)
            rr, cc = min(n, hi), min(n, wi)
            new[:rr, :cc] = I[:rr, :cc]
            box[0] = new
            snap()

        order = diag if downward else diag[::-1]

        if not np.array_equal(box[0][:n, :n], target):
            if prime_first:
                fill_rect(0, 0, n, n, bgc)              # lay the background sheet down first
                if style == 2:
                    paint([(0, 0)], fgc)                # the first step of the stair
                    m = 1
                    while m < n:                        # then keep copying the stair onto itself
                        k = min(m, n - m)
                        g = box[0]
                        clip = g[0:k, 0:k].copy()
                        dest = g[m:m + k, m:m + k]
                        if np.any((clip != 0) & (dest != clip)):
                            ops.append(29)
                            sels.append([0, 0, k - 1, k - 1])          # copy the stair built so far
                            ops.append(30)
                            sels.append([m, m, 0, 0])                  # repeat it further down
                            dest[clip != 0] = clip[clip != 0]
                            snap()
                        left = [(m + i, m + j) for i in range(k) for j in range(k)
                                if clip[i, j] == 0 and dest[i, j] != 0]
                        if left:
                            paint(left, 0)   # colour 0 never travels with a Paste - put it down by hand
                        m += k
                elif style == 1:
                    for rc in order:
                        paint([rc], fgc)                # one step of the stair at a time
                else:
                    paint(diag, fgc)                    # the whole diagonal in one stroke
            else:
                if style == 1:
                    for rc in order:
                        paint([rc], fgc)
                else:
                    paint(diag, fgc)
                paint(off, bgc)                         # then everything that is not the diagonal

        if crop_last and (hi, wi) != (n, n):
            ops.append(33)
            sels.append([0, 0, n - 1, n - 1])
            box[0] = box[0][:n, :n].copy()
            snap()

        ops.append(34)
        sels.append([0, 0, n - 1, n - 1])
        return ops, sels, states, box[0]

    # ---- route choices, all read off I: sheet size vs picture size, object count,
    #      how many cells the picture uses, where its first object sits ----
    crop_last = (n <= hi and n <= wi) and ((hi + wi) % 2 == 0)
    prime_first = (n % 2 == 0)
    downward = ((first[0] + first[1]) % 2 == 0)
    style = nfg % 3                      # 0 = one stroke, 1 = step by step, 2 = copy the stair
    if style == 2 and (n < 4 or fgc == 0):
        style = 0                        # a 0-coloured stair cannot be copied: draw it instead
    if style == 2:
        prime_first = True               # the stair can only be copied off a finished background

    plans = [(style, prime_first, downward, crop_last), (0, True, True, False)]
    ops, sels = [], []
    for st, pf, dw, cl in plans:
        ops, sels, states, final = build(st, pf, dw, cl)
        if np.array_equal(final, target) and len(set(states)) == len(states):
            break
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
                        f"num_examples+1 ({num_examples + 1}) for task d0f5fe59"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d0f5fe59"
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
                                f"for task d0f5fe59"
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
                    f"Failed to build a complete episode for task d0f5fe59 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d0f5fe59-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
