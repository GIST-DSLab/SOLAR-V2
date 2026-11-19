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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# How this differs from the maker: the code-golf entry treats each dot as one unit
# that is lifted from its cell and set down beside the line of its own colour.
# The maker painted the old cell to background and the new cell with the colour.
# Here each matching dot is grabbed once and slid with Move ops until it sits next
# to its line. The two readings agree on the noise dots: dots whose colour has no
# line just disappear. Here that is one Color(bgc) per noise colour.


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    transpose = random.choice((True, False))
    H, W = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (8, max(8, min(30, H))))
    w = _unifint(diff_lb, diff_ub, (8, max(8, min(30, W))))
    remcols = [c for c in cols if c != bgc]
    nlines = _unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = random.sample(remcols, nlines)
    remcols = [c for c in remcols if c not in linecols]
    nnoisecols = _unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = random.sample(remcols, nnoisecols)
    locopts = list(range(w))
    locs = []
    for _ in range(nlines):
        if not locopts:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if not (loc - 2 <= x < loc + 3)]
        locs.append(loc)
    locs = sorted(locs)
    linecols = linecols[:len(locs)]
    nlines = len(locs)
    gi = [[bgc] * w for _ in range(h)]
    for loc, col in zip(locs, linecols):
        for r in range(h):
            gi[r][loc] = col
    go = [row[:] for row in gi]
    nilocs = _unifint(diff_lb, diff_ub, (1, h))
    ilocs = random.sample(range(h), nilocs)
    dotlocopts = [c for c in range(w) if c not in locs]
    pool = sorted(set(linecols) | set(noisecols))
    for ii in ilocs:
        ub = min(nlines + nnoisecols, (w - nlines) // 2 - 1)
        ndots = _unifint(diff_lb, diff_ub, (1, max(1, ub)))
        dotlocs = random.sample(dotlocopts, ndots)
        dotcols = random.sample(pool, ndots)
        for j, col in zip(dotlocs, dotcols):
            gi[ii][j] = col
            if col in linecols:
                L = locs[linecols.index(col)]
                go[ii][L + 1 if j > L else L - 1] = col
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the canvas colour (lines take at most ~1/5 of the grid, dots are sparse)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # frontiers: full rows / full columns of one non-background colour
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

    ops, sels = [], []
    G = I.copy()

    # 1. dots whose colour has no line vanish: one Color(bgc) per noise colour
    noise_cols = sorted({int(I[p]) for p in dots if int(I[p]) not in line_of})
    for k in noise_cols:
        cells = [p for p in dots if int(I[p]) == k]
        ops.append(bgc); sels.append(sel_of(cells))
        for p in cells:
            G[p] = bgc

    # 2. each matching dot gravitates toward its line (one unit = one dot)
    def landing(p, k):
        o, idx = line_of[k]
        r, c = p
        if o == 'row':
            return (idx + 1 if r > idx else idx - 1, c)
        return (r, idx + 1 if c > idx else idx - 1)

    movers = []
    for p in dots:
        k = int(I[p])
        if k in line_of:
            dst = landing(p, k)
            if dst != p:
                movers.append((p, dst, k))

    # nearest-to-line first, line by line
    def dist(m):
        return abs(m[0][0] - m[1][0]) + abs(m[0][1] - m[1][1])
    movers.sort(key=lambda m: (line_of[m[2]], dist(m), m[0]))

    pending = list(movers)
    destroyed = set()  # dot sources overwritten in a landing cycle

    def do_move(src, dst, k):
        dr = (dst[0] > src[0]) - (dst[0] < src[0])
        dc = (dst[1] > src[1]) - (dst[1] < src[1])
        if k == 0:
            # 0 is "nothing" to ARCLE object ops: a 0-dot cannot be grabbed
            ops.append(bgc); sels.append(sel_of([src])); G[src] = bgc
            if G[dst] != 0:
                ops.append(0); sels.append(sel_of([dst])); G[dst] = 0
            return
        steps = abs(dst[0] - src[0]) + abs(dst[1] - src[1])
        op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(op); sels.append(sel_of([src]))          # grab the dot once
        for _ in range(steps - 1):
            ops.append(op); sels.append(sel_of([]))         # keep sliding the same dot
        G[dst] = k
        G[src] = 0
        # ARCLE leaves the vacated start cell at 0; that is already right when a
        # still-pending 0-coloured dot is due to land exactly there
        zero_lands = {m[1] for m in pending if m[2] == 0}
        if bgc != 0 and src not in zero_lands:
            ops.append(bgc); sels.append(sel_of([src]))
            G[src] = bgc

    while pending:
        srcs = {m[0] for m in pending if m[0] not in destroyed}
        pick = None
        for m in pending:
            if m[1] not in srcs:      # landing cell not occupied by an unmoved dot
                pick = m
                break
        if pick is None:              # cycle of dots sitting on each other's landings
            pick = pending[0]
        pending.remove(pick)
        src, dst, k = pick
        if src in destroyed:
            # its cell was taken by another dot's landing; it reappears at its own
            if G[dst] != k:
                ops.append(k); sels.append(sel_of([dst])); G[dst] = k
            continue
        if dst in srcs:
            destroyed.add(dst)
        do_move(src, dst, k)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
