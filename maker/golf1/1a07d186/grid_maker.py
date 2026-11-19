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


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is fixed for the episode. The rule depends only on
    # whether a dot's colour matches a line's colour, not on which colours those are.
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]  # vertical lines or horizontal lines
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None) -> dict:
    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lb + (ub - lb) * d))

    if mirror is None:
        mirror = random.choice((True, False))
    cols = list(range(10))
    # when mirrored the grid is transposed, so the pre-transpose h/w are swapped bounds
    Hmax, Wmax = (max_w, max_h) if mirror else (max_h, max_w)
    h = unifint(8, max(8, min(30, Hmax)))
    w = unifint(8, max(8, min(30, Wmax)))
    remcols = [c for c in cols if c != bgc]
    nlines = unifint(1, w // 5)
    linecols = random.sample(remcols, nlines)
    remcols = [c for c in remcols if c not in linecols]
    nnoisecols = unifint(0, len(remcols))
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
    gi = [[bgc] * w for _ in range(h)]
    for loc, col in zip(locs, linecols):
        for r in range(h):
            gi[r][loc] = col
    go = [row[:] for row in gi]
    nilocs = unifint(1, h)
    ilocs = random.sample(range(h), nilocs)
    dotlocopts = [x for x in range(w) if x not in locs]
    allcols = sorted(set(linecols) | set(noisecols))
    for ii in ilocs:
        ub = max(1, min(nlines + nnoisecols, (w - nlines) // 2 - 1))
        ndots = unifint(1, ub)
        ndots = min(ndots, len(dotlocopts), len(allcols))
        dotlocs = random.sample(dotlocopts, ndots)
        dotcols = random.sample(allcols, ndots)
        for dj, col in zip(dotlocs, dotcols):
            gi[ii][dj] = col
            if col in linecols:
                lj = locs[linecols.index(col)]
                if dj > lj:
                    go[ii][lj + 1] = col
                else:
                    go[ii][lj - 1] = col
    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if mirror:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Lines: whole rows or columns of one colour that is not the background.
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
    for orient, idx, _ in lines:
        if orient == 'row':
            on_line[idx, :] = True
        else:
            on_line[:, idx] = True
    line_of_color = {}
    for orient, idx, k in lines:
        line_of_color.setdefault(k, (orient, idx))

    # Dots: every non-background cell that is not part of a line.
    dots = [(int(r), int(c)) for r, c in zip(*np.nonzero((I != bgc) & (~on_line)))]

    ops, sels = [], []

    # 1. Clear every dot off the canvas in one stroke (noise and matching dots alike).
    if dots:
        ops.append(int(bgc)); sels.append(sel_of(dots))

    # 2. For each line, paint in one stroke the cells its matching dots slide to:
    #    the cell next to the line, in the dot's own row/column, on the dot's side.
    for orient, idx, k in sorted(lines, key=lambda t: (t[0], t[1])):
        if line_of_color.get(k) != (orient, idx):
            continue
        land = set()
        for (r, c) in dots:
            if int(I[r, c]) != k:
                continue
            if orient == 'row':
                land.add((idx + 1 if r > idx else idx - 1, c))
            else:
                land.add((r, idx + 1 if c > idx else idx - 1))
        if land:
            ops.append(int(k)); sels.append(sel_of(sorted(land)))

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
