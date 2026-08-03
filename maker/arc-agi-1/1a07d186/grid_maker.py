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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    do_mirror = choice((True, False))
    hcap = max_w if do_mirror else max_h
    wcap = max_h if do_mirror else max_w
    h = unifint(diff_lb, diff_ub, (8, max(8, hcap)))
    w = unifint(diff_lb, diff_ub, (8, max(8, wcap)))
    remcols = remove(bgc, cols)
    nlines = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
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
        ub = min(nlines + nnoisecols, (w - nlines) // 2 - 1)
        if ub < 1:
            continue
        ndots = unifint(diff_lb, diff_ub, (1, ub))
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
    if do_mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 1. rule discovery from I: full-grid uniform lines (frontiers) ---
    lines = []  # (orient, index, color)
    for r in range(h):
        v = I[r, 0]
        if v != bgc and np.all(I[r] == v):
            lines.append(('row', r, int(v)))
    for c in range(w):
        v = I[0, c]
        if v != bgc and np.all(I[:, c] == v):
            lines.append(('col', c, int(v)))

    on_line = np.zeros((h, w), dtype=bool)
    for orient, idx, _ in lines:
        if orient == 'row':
            on_line[idx, :] = True
        else:
            on_line[:, idx] = True

    line_of_color = {}
    for orient, idx, k in lines:
        line_of_color.setdefault(k, (orient, idx))

    # --- 2. dots = non-background cells not belonging to a line ---
    dots = [(int(r), int(c)) for r, c in zip(*np.nonzero((I != bgc) & (~on_line)))]

    def landing_of(r, c, k):
        orient, idx = line_of_color[k]
        if orient == 'row':
            return (idx + 1 if r > idx else idx - 1, c)
        return (r, idx + 1 if c > idx else idx - 1)

    # --- 3. target grid derived from the rule (dots cleared, matches gravitate) ---
    T = I.copy()
    for (r, c) in dots:
        T[r, c] = bgc
    for (r, c) in dots:
        k = int(I[r, c])
        if k in line_of_color:
            lr, lc = landing_of(r, c, k)
            T[lr, lc] = k

    ops, sels = [], []
    Wg = I.copy()

    # --- 4. remove noise dots (colors with no matching line), component by component ---
    noise_set = set((r, c) for (r, c) in dots if int(I[r, c]) not in line_of_color)
    seen = set()
    for (r, c) in sorted(noise_set):
        if (r, c) in seen:
            continue
        col = int(I[r, c])
        comp, stack = [], [(r, c)]
        seen.add((r, c))
        while stack:
            a, b = stack.pop()
            comp.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (a + da, b + db)
                if p in noise_set and p not in seen and int(I[p[0], p[1]]) == col:
                    seen.add(p)
                    stack.append(p)
        if all(T[p[0], p[1]] == bgc for p in comp):
            ops.append(10 + int(bgc))          # FloodFill<bgc> wipes the whole noise blob
            sels.append([r, c, 0, 0])
            for p in comp:
                Wg[p[0], p[1]] = bgc
        else:
            for p in comp:                      # blob overlaps a landing cell: clear only free cells
                if T[p[0], p[1]] == bgc and Wg[p[0], p[1]] != bgc:
                    ops.append(int(bgc))
                    sels.append([p[0], p[1], 0, 0])
                    Wg[p[0], p[1]] = bgc

    # --- 5. per line: its matching dots gravitate onto it, nearest first ---
    for orient, idx, k in sorted(lines, key=lambda t: (t[0], t[1])):
        if line_of_color.get(k) != (orient, idx):
            continue
        mine = [(r, c) for (r, c) in dots if int(I[r, c]) == k]
        mine.sort(key=lambda p: (abs(p[0] - idx) if orient == 'row' else abs(p[1] - idx), p))
        for (r, c) in mine:
            lr, lc = landing_of(r, c, k)
            if T[r, c] == bgc and Wg[r, c] != bgc:
                ops.append(int(bgc))            # dot leaves its old cell
                sels.append([r, c, 0, 0])
                Wg[r, c] = bgc
            if Wg[lr, lc] != k:
                ops.append(int(k))              # dot arrives next to its line
                sels.append([lr, lc, 0, 0])
                Wg[lr, lc] = k

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
