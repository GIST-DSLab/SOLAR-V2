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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    hmax, wmax = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (8, max(8, hmax)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, wmax)))
    remcols = [c for c in range(10) if c != bgc]
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
        locopts = [x for x in locopts if not (loc - 2 <= x <= loc + 2)]
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = np.full((h, w), bgc, dtype=int)
    for loc, col in zip(locs, linecols):
        gi[:, loc] = col
    go = gi.copy()
    nilocs = _unifint(diff_lb, diff_ub, (1, h))
    ilocs = random.sample(range(h), nilocs)
    dotlocopts = [x for x in range(w) if x not in locs]
    allcols = sorted(set(linecols) | set(noisecols))
    for ii in ilocs:
        ub = min(nlines + nnoisecols, (w - nlines) // 2 - 1)
        ndots = _unifint(diff_lb, diff_ub, (1, max(1, ub)))
        ndots = min(ndots, len(allcols), len(dotlocopts))
        dotlocs = random.sample(dotlocopts, ndots)
        dotcols = random.sample(allcols, ndots)
        for dj, col in zip(dotlocs, dotcols):
            gi[ii, dj] = col
            if col in linecols:
                lj = locs[linecols.index(col)]
                go[ii, lj + 1 if dj > lj else lj - 1] = col
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Bars: full-length uniform non-background columns (or rows if transposed).
    col_bars = [(c, int(I[0, c])) for c in range(w)
                if I[0, c] != bgc and np.all(I[:, c] == I[0, c])]
    row_bars = [(r, int(I[r, 0])) for r in range(h)
                if I[r, 0] != bgc and np.all(I[r, :] == I[r, 0])]
    vertical = len(col_bars) >= len(row_bars)
    bars = col_bars if vertical else row_bars
    on_bar = np.zeros((h, w), dtype=bool)
    for idx, _ in bars:
        if vertical:
            on_bar[:, idx] = True
        else:
            on_bar[idx, :] = True
    bar_of = {}
    for idx, k in bars:
        bar_of.setdefault(k, idx)

    pixels = [(int(r), int(c)) for r, c in zip(*np.nonzero((I != bgc) & ~on_bar))]
    ops, sels = [], []
    G = I.copy()

    # 1. Pixels whose colour has no bar are deleted: one Color(bgc) per such colour.
    noise_cols = sorted({int(I[p]) for p in pixels if int(I[p]) not in bar_of})
    for col in noise_cols:
        cells = [p for p in pixels if int(I[p]) == col]
        ops.append(int(bgc)); sels.append(sel_of(cells))
        for p in cells:
            G[p] = bgc

    # 2. Each matching pixel slides along its row/column until it touches its bar.
    movers = {}
    for p in pixels:
        k = int(I[p])
        if k not in bar_of:
            continue
        idx = bar_of[k]
        pos = p[1] if vertical else p[0]
        land = idx + 1 if pos > idx else idx - 1
        dest = (p[0], land) if vertical else (land, p[1])
        if dest != p:
            movers[p] = dest

    # order: a pixel sitting on another pixel's landing cell must leave first;
    # within that constraint, nearest-to-bar first.
    remaining = dict(movers)
    order = []
    while remaining:
        # a pixel may go once its landing cell holds no still-pending pixel
        free = [p for p in remaining if remaining[p] not in remaining]
        if not free:
            free = [next(iter(sorted(remaining)))]  # cycle: rare fallback
        free.sort(key=lambda p: (abs((p[1] if vertical else p[0]) -
                                     (remaining[p][1] if vertical else remaining[p][0])), p))
        p = free[0]
        order.append((p, remaining.pop(p)))

    for src, dest in order:
        col = int(I[src])
        if G[src] != col or col == 0:
            # colour 0 cannot be grabbed by Move (0 = empty), and a pixel overwritten
            # in a rare swap cycle is gone: erase the source (if still there) and
            # paint the pixel where it lands against its bar.
            if G[src] == col:
                ops.append(int(bgc)); sels.append(sel_of([src]))
                G[src] = bgc
            ops.append(col); sels.append(sel_of([dest]))
            G[dest] = col
            continue
        dr, dc = dest[0] - src[0], dest[1] - src[1]
        n = abs(dr) + abs(dc)
        if dr:
            op = 21 if dr > 0 else 20
        else:
            op = 22 if dc > 0 else 23
        # grab the pixel once, then continue the slide with empty selections
        ops.append(op); sels.append(sel_of([src]))
        for _ in range(n - 1):
            ops.append(op); sels.append(sel_of([]))
        G[src] = 0
        G[dest] = col
        if bgc != 0:
            # repair only the pixel's original footprint
            ops.append(int(bgc)); sels.append(sel_of([src]))
            G[src] = bgc

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
