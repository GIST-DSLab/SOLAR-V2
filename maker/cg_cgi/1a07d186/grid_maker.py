"""
ARC Task: 1a07d186 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

VARIANTS = [{"transpose": False}, {"transpose": True}]

def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h, max_w, bgc, transpose=None, **kw) -> dict:
    from random import sample, choice
    from dsl import (interval, remove, difference, canvas, fill,
                     connect, totuple, dmirror)
    from utils import unifint
    if transpose is None:
        transpose = random.choice([False, True])
    cols = interval(0, 10, 1)
    # after dmirror h/w swap; bound accordingly
    hb, wb = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (8, max(8, hb)))
    w = unifint(diff_lb, diff_ub, (8, max(8, wb)))
    remcols = remove(bgc, cols)
    nlines = unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = sample(remcols, nlines)
    remcols = difference(remcols, linecols)
    nnoisecols = unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    locopts = interval(0, w, 1)
    locs = []
    for k in range(nlines):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = difference(locopts, interval(loc - 2, loc + 3, 1))
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = canvas(bgc, (h, w))
    for loc, col in zip(locs, linecols):
        gi = fill(gi, col, connect((0, loc), (h - 1, loc)))
    go = tuple(e for e in gi)
    nilocs = unifint(diff_lb, diff_ub, (1, h))
    ilocs = sample(interval(0, h, 1), nilocs)
    dotlocopts = difference(interval(0, w, 1), locs)
    for ii in ilocs:
        ndots = unifint(diff_lb, diff_ub, (1, min(nlines + nnoisecols, (w - nlines) // 2 - 1)))
        dotlocs = sample(dotlocopts, ndots)
        dotcols = sample(totuple(set(linecols) | set(noisecols)), ndots)
        for dotlocj, col in zip(dotlocs, dotcols):
            gi = fill(gi, col, {(ii, dotlocj)})
            if col in linecols:
                idx = linecols.index(col)
                linelocj = locs[idx]
                if dotlocj > linelocj:
                    go = fill(go, col, {(ii, linelocj + 1)})
                else:
                    go = fill(go, col, {(ii, linelocj - 1)})
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}

def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # background: the canvas colour (dots fill at most half a row, so it dominates)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 1. the frontiers: full-length uniform rows / columns
    lines = []
    for r in range(h):
        v = int(I[r, 0])
        if v != bgc and np.all(I[r] == v):
            lines.append(('row', r, v))
    for c in range(w):
        v = int(I[0, c])
        if v != bgc and np.all(I[:, c] == v):
            lines.append(('col', c, v))
    on_line = np.zeros((h, w), dtype=bool)
    line_of = {}
    for o, idx, k in lines:
        if o == 'row':
            on_line[idx, :] = True
        else:
            on_line[:, idx] = True
        line_of.setdefault(k, (o, idx))

    dots = [(int(r), int(c)) for r, c in zip(*np.nonzero((I != bgc) & ~on_line))]

    def landing(r, c, k):
        o, idx = line_of[k]
        if o == 'row':
            return (idx + 1 if r > idx else idx - 1, c)
        return (r, idx + 1 if c > idx else idx - 1)

    def dist(p):
        o, idx = line_of[int(I[p])]
        return abs(p[0] - idx) if o == 'row' else abs(p[1] - idx)

    ops, sels = [], []
    W = I.copy()

    # 2. dots whose colour owns no line vanish: wipe each such blob
    noise = set(p for p in dots if int(I[p]) not in line_of)
    seen = set()
    for p in sorted(noise):
        if p in seen:
            continue
        col = int(I[p])
        comp, st = [], [p]
        seen.add(p)
        while st:
            a, b = st.pop()
            comp.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (a + da, b + db)
                if q in noise and q not in seen and int(I[q]) == col:
                    seen.add(q)
                    st.append(q)
        ops.append(int(bgc)); sels.append(sel_of(comp))
        for q in comp:
            W[q] = bgc

    # 3. matching dots slide perpendicular onto their line (nearest first;
    #    a dot sitting on another dot's landing cell leaves before it is needed)
    movers = [p for p in dots if int(I[p]) in line_of and landing(p[0], p[1], int(I[p])) != p]
    movers.sort(key=lambda p: (dist(p), p))
    pending = list(movers)
    while pending:
        pick = None
        for q in pending:
            others = {landing(x[0], x[1], int(I[x])) for x in pending if x != q}
            if q not in others:
                pick = q
                break
        if pick is None:
            pick = pending[0]
        pending.remove(pick)
        r, c = pick
        k = int(I[pick])
        lr, lc = landing(r, c, k)
        if W[r, c] != k or k == 0:
            # colour-0 dot (Move carries only nonzero cells) or a dot already
            # overwritten in a swap cycle: lift it off / set it down by painting
            if W[r, c] == k:
                ops.append(int(bgc)); sels.append(sel_of([(r, c)])); W[r, c] = bgc
            if W[lr, lc] != k:
                ops.append(k); sels.append(sel_of([(lr, lc)])); W[lr, lc] = k
            continue
        dr = (lr > r) - (lr < r)
        dc = (lc > c) - (lc < c)
        op = 21 if dr > 0 else 20 if dr < 0 else 22 if dc > 0 else 23
        steps = abs(lr - r) + abs(lc - c)
        snap = W.copy(); snap[r, c] = 0
        ops.append(op); sels.append(sel_of([(r, c)]))      # grab the dot
        for _ in range(steps - 1):
            ops.append(op); sels.append(sel_of([]))        # keep sliding same dot
        W = snap.copy(); W[lr, lc] = k
        if bgc != 0:
            ops.append(int(bgc)); sels.append(sel_of([(r, c)])); W[r, c] = bgc

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
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
                                f"for task 1a07d186"
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
                    f"Failed to build a complete episode for task 1a07d186 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1a07d186-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
